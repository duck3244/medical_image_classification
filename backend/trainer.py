"""
trainer.py
2단계 학습 루프 (head-only → fine-tune) + Focal Loss + Gradient Accumulation.

핵심 특징:
- Keras fit()을 기본으로 사용하되, grad_accum_steps>1일 때만 커스텀 train_step 경유
- Binary / Categorical 공용 Focal Loss
- OOM 발생 시 batch_size 절반으로 자동 재시도 (최대 2회)
- 콜백: ModelCheckpoint(val AUC 기준), EarlyStopping, ReduceLROnPlateau, TensorBoard
- 학습 종료 후 history.json + best weight 경로 반환
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as kc
from tensorflow.keras import optimizers as ko

from model import build_model, unfreeze_for_finetune, count_params
from data_processor import build_datasets
from utils import get_task_params, log_vram_usage, save_json


# ======================================================================
# Focal Loss
# ======================================================================

def binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """sigmoid 출력용 binary focal loss."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        at = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(-at * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss_fn


def categorical_focal_loss(alpha=0.25, gamma: float = 2.0):
    """softmax 출력용 categorical focal loss (one-hot 라벨).

    alpha:
        - float: 모든 클래스에 동일 가중치 (기존 동작과 호환)
        - 1D 시퀀스/array: 클래스별 가중치 벡터. 길이는 num_classes와 일치해야 한다.
          ISIC 같이 분포가 심하게 치우친 다중분류에서 소수 클래스에 더 큰 가중치를
          주려면 벡터 형태를 사용한다.
    """
    if hasattr(alpha, "__len__") and not isinstance(alpha, (str, bytes)):
        alpha_t = tf.constant(list(alpha), dtype=tf.float32)
    else:
        alpha_t = tf.constant(float(alpha), dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        # alpha_t가 벡터면 클래스별 가중치를 axis=-1 브로드캐스트로 적용.
        weight = alpha_t * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn


def build_loss(is_binary: bool, kind: str, alpha, gamma: float):
    if kind == "focal":
        return binary_focal_loss(alpha, gamma) if is_binary else categorical_focal_loss(alpha, gamma)
    if is_binary:
        return tf.keras.losses.BinaryCrossentropy()
    return tf.keras.losses.CategoricalCrossentropy()


# ======================================================================
# Gradient Accumulation (커스텀 Model 래퍼)
# ======================================================================

class GradAccumModel(tf.keras.Model):
    """grad_accum_steps 만큼 gradient를 누적 후 한 번에 apply.

    제약: 단일 GPU 환경 전용. tf.distribute strategy 하에서는
    accumulator counter가 replica별로 분기하여 sync가 어그러질 수 있으므로
    분산 학습 시 사용을 금지한다 (생성자에서 검사).
    """

    def __init__(self, inner: tf.keras.Model, accum_steps: int = 1):
        super().__init__()
        # 분산 학습 환경 차단: replica 수가 1이 아니면 명시적 에러.
        try:
            num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        except Exception:
            num_replicas = 1
        if num_replicas > 1:
            raise RuntimeError(
                f"GradAccumModel은 단일 GPU 전용이지만 num_replicas_in_sync={num_replicas}. "
                "분산 학습 시에는 grad_accum_steps=1로 설정하거나 "
                "tf.distribute 호환 누적 구현을 사용하세요."
            )
        self.inner = inner
        self.accum_steps = max(1, int(accum_steps))
        self._accum_grads = None
        self._accum_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

    def call(self, inputs, training=False):
        return self.inner(inputs, training=training)

    def _ensure_accum(self):
        tvars = self.inner.trainable_variables
        need_init = (
            self._accum_grads is None
            or len(self._accum_grads) != len(tvars)
        )
        if need_init:
            self._accum_grads = [
                tf.Variable(tf.zeros_like(v), trainable=False, name=f"accum_{i}")
                for i, v in enumerate(tvars)
            ]
            self._accum_counter.assign(0)

    def reset_accumulators(self):
        """단계 전환 시 호출 — 누적 변수 강제 재생성."""
        self._accum_grads = None
        self._accum_counter.assign(0)

    def _apply_and_reset(self):
        self.optimizer.apply_gradients(
            zip(self._accum_grads, self.inner.trainable_variables)
        )
        for ag in self._accum_grads:
            ag.assign(tf.zeros_like(ag))
        self._accum_counter.assign(0)
        return tf.constant(0)

    def _no_op(self):
        return tf.constant(0)

    def flush_accumulated(self):
        """Epoch 경계에서 남은 부분 누적 gradient를 적용.
        loss는 /accum_steps로 스케일된 상태이므로 실제 누적 건수(counter)로 재스케일.
        """
        if self._accum_grads is None:
            return
        counter = int(self._accum_counter.numpy())
        if counter <= 0:
            return
        scale = float(self.accum_steps) / float(counter)
        for ag in self._accum_grads:
            ag.assign(ag * scale)
        self._apply_and_reset()

    def train_step(self, data):
        x, y = data
        self._ensure_accum()
        with tf.GradientTape() as tape:
            y_pred = self.inner(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            scaled_loss = loss / tf.cast(self.accum_steps, loss.dtype)

        grads = tape.gradient(scaled_loss, self.inner.trainable_variables)
        for ag, g in zip(self._accum_grads, grads):
            if g is not None:
                ag.assign_add(g)

        self._accum_counter.assign_add(1)
        tf.cond(
            tf.equal(self._accum_counter, self.accum_steps),
            self._apply_and_reset,
            self._no_op,
        )

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.inner(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# ======================================================================
# 메트릭 / 콜백 / 옵티마이저
# ======================================================================

def build_metrics(is_binary: bool):
    if is_binary:
        return [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    return [
        tf.keras.metrics.CategoricalAccuracy(name="acc"),
        # 단일 라벨 다중분류 → multi_label=False (OvR 평균). evaluator의
        # sklearn roc_auc_score(multi_class='ovr')와 정합.
        tf.keras.metrics.AUC(name="auc", multi_label=False),
    ]


def build_optimizer(name: str, lr: float, logger=None):
    """이름 기반 옵티마이저 생성.

    AdamW 요청 시: tf.keras → tensorflow_addons → 평문 Adam 순으로 폴백.
    평문 Adam으로 폴백되면 weight_decay가 적용되지 않으므로 명시적으로 경고한다.
    """
    name = (name or "adamw").lower()
    if name == "adamw":
        wd = 1e-3
        try:
            return ko.AdamW(learning_rate=lr, weight_decay=wd)
        except AttributeError:
            try:
                from tensorflow_addons.optimizers import AdamW
                return AdamW(weight_decay=wd, learning_rate=lr)
            except ImportError:
                msg = (
                    "AdamW 미지원 (tf.keras.optimizers.AdamW 없음 AND tensorflow_addons 미설치). "
                    "weight_decay가 적용되지 않는 평문 Adam으로 폴백합니다. "
                    "복구: pip install tensorflow-addons==0.18.0"
                )
                if logger is not None:
                    logger.warning(msg)
                else:
                    import warnings
                    warnings.warn(msg, RuntimeWarning)
                return ko.Adam(learning_rate=lr)
    if name == "sgd":
        return ko.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    return ko.Adam(learning_rate=lr)


class GradAccumFlushCallback(kc.Callback):
    """매 epoch 끝에 GradAccumModel의 잔여 gradient를 flush."""
    def __init__(self, accum_model: "GradAccumModel"):
        super().__init__()
        self._accum_model = accum_model

    def on_epoch_end(self, epoch, logs=None):
        try:
            self._accum_model.flush_accumulated()
        except Exception as e:
            print(f"[GradAccumFlush] epoch {epoch+1} flush 실패: {e}")


class InnerWeightsCheckpoint(kc.Callback):
    """GradAccumModel 래퍼 우회 — inner 모델의 weights만 저장."""
    def __init__(self, inner, filepath, monitor, mode="max"):
        super().__init__()
        self.inner = inner
        self.filepath = str(filepath)
        self.monitor = monitor
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if v is None:
            return
        improved = v > self.best if self.mode == "max" else v < self.best
        if improved:
            self.best = v
            Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
            self.inner.save_weights(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {v:.5f}, "
                  f"saving inner weights to {self.filepath}")


def build_callbacks(
    run_dir: Path,
    monitor: str,
    es_patience: int,
    lr_patience: int,
    inner_for_ckpt: Optional[tf.keras.Model] = None,
) -> list:
    ckpt_path = run_dir / "best.weights.h5"
    if inner_for_ckpt is not None:
        ckpt_cb = InnerWeightsCheckpoint(
            inner_for_ckpt, ckpt_path, monitor=monitor, mode="max"
        )
    else:
        ckpt_cb = kc.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    return [
        ckpt_cb,
        kc.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        kc.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        kc.TensorBoard(log_dir=str(run_dir / "tb"), histogram_freq=0),
        kc.CSVLogger(str(run_dir / "history.csv")),
    ]


# ======================================================================
# 학습 파이프라인
# ======================================================================

def _wrap_for_accum(inner: tf.keras.Model, accum_steps: int) -> tf.keras.Model:
    if accum_steps <= 1:
        return inner
    return GradAccumModel(inner, accum_steps=accum_steps)


def _resolve_fine_tune_at(cfg_value, backbone: str, logger) -> int:
    """fine_tune_at 설정을 백본에 맞는 정수로 해석.
    - int: 그대로 사용 (legacy)
    - dict[backbone -> int]: backbone 키로 조회. 키 누락 시 명시적으로 raise.
    """
    if isinstance(cfg_value, dict):
        if backbone not in cfg_value:
            raise KeyError(
                f"config.fine_tune_at 에 backbone '{backbone}' 키가 없습니다. "
                f"존재 키: {list(cfg_value.keys())}"
            )
        v = int(cfg_value[backbone])
        logger.info(f"fine_tune_at[{backbone}] = {v}")
        return v
    return int(cfg_value)


def _compile(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train(config: dict, logger) -> dict:
    """
    전체 학습 파이프라인. config은 config.json 로드 결과.
    Returns: {"best_weights": str, "history": {...}, "meta": {...}}
    """
    tp = get_task_params(config)
    is_binary = tp["is_binary"]
    accum_steps = int(config.get("grad_accum_steps", 1))

    # 1) 데이터셋
    batch_size = int(config["batch_size"])
    train_ds, val_ds, test_ds, meta = _build_train_datasets(config, tp, batch_size)
    logger.info(f"dataset sizes: {meta['sizes']}")
    logger.info(f"class weights: {meta['class_weights']}")

    # 2) 모델
    inner, base = build_model(
        backbone=config["backbone"],
        num_classes=tp["num_classes"],
        img_size=tp["img_size"],
        dropout=float(config.get("dropout", 0.5)),
        l2_reg=float(config.get("l2_reg", 1e-4)),
    )
    logger.info(f"params (stage1): {count_params(inner)}")

    # 3) 실행 디렉토리
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"]["models_dir"]) / f"{tp['task']}_{config['backbone']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {"config": config, "meta": meta, "task_params": tp},
        str(run_dir / "run_config.json"),
    )

    # Focal alpha 자동화:
    # - binary: 양성 비율 기반 스칼라 alpha (alpha = 1 - pos_ratio).
    # - multiclass: balanced class_weight를 sum=num_classes로 정규화한 per-class 벡터.
    #   ISIC 같이 분포가 한쪽으로 치우친 경우 스칼라 alpha는 효과가 미미하므로 벡터로 가중.
    focal_alpha = float(config.get("focal_alpha", 0.25))
    if config.get("loss") == "focal" and bool(config.get("focal_alpha_auto", False)):
        cw = meta.get("class_weights", {})
        if is_binary:
            if 0 in cw and 1 in cw:
                # balanced class_weight = total / (2*count_c). pos_ratio = w0/(w0+w1).
                pos_ratio = cw[0] / (cw[0] + cw[1])
                focal_alpha = float(max(0.05, min(0.95, 1.0 - pos_ratio)))
                logger.info(
                    f"focal_alpha auto-set to {focal_alpha:.4f} (pos_ratio={pos_ratio:.4f})"
                )
        else:
            n_cls = tp["num_classes"]
            if all(i in cw for i in range(n_cls)):
                raw = np.array([cw[i] for i in range(n_cls)], dtype=np.float32)
                # sum=n_cls로 정규화 → 클래스 평균 alpha가 1.0이 되도록 스케일.
                alpha_vec = raw * (n_cls / raw.sum())
                # 극단치 방지 클램프
                alpha_vec = np.clip(alpha_vec, 0.05, 10.0)
                focal_alpha = alpha_vec.tolist()
                logger.info(
                    "focal_alpha auto-set (per-class): "
                    + ", ".join(f"{tp['class_names'][i]}={v:.3f}"
                                for i, v in enumerate(alpha_vec))
                )

    loss = build_loss(
        is_binary,
        config.get("loss", "focal"),
        focal_alpha,
        float(config.get("focal_gamma", 2.0)),
    )
    metrics = build_metrics(is_binary)
    monitor = f"val_{config.get('monitor_metric', 'auc').replace('val_', '')}"

    # 클래스 가중치 (이진 Focal Loss에는 보통 중복이라 skip, CE에만 적용)
    use_class_weight = (
        bool(config.get("class_weight_auto", True))
        and config.get("loss") != "focal"
    )
    class_weight = meta["class_weights"] if use_class_weight else None

    # =========================
    # Stage 1: head-only
    # =========================
    logger.info("=" * 60)
    logger.info("Stage 1: head-only training")
    logger.info("=" * 60)
    model = _wrap_for_accum(inner, accum_steps)
    _compile(model, build_optimizer(config["optimizer"], float(config["lr_head"]), logger), loss, metrics)

    def _make_cbs_stage1():
        out = build_callbacks(
            run_dir / "stage1",
            monitor=monitor,
            es_patience=int(config["early_stopping_patience"]),
            lr_patience=int(config["reduce_lr_patience"]),
            inner_for_ckpt=inner if accum_steps > 1 else None,
        )
        if isinstance(model, GradAccumModel):
            out.append(GradAccumFlushCallback(model))
        return out

    (run_dir / "stage1").mkdir(parents=True, exist_ok=True)
    cbs = _make_cbs_stage1()

    hist1, train_ds, val_ds, batch_size = _fit_with_oom_retry(
        model,
        train_ds=train_ds, val_ds=val_ds,
        epochs=int(config["epochs_head"]),
        callbacks=cbs,
        class_weight=class_weight,
        stage_name="Stage1",
        current_batch=batch_size,
        config=config, tp=tp,
        callbacks_builder=_make_cbs_stage1,
        logger=logger,
    )
    log_vram_usage(logger, "after stage1")

    # =========================
    # Stage 2: fine-tune
    # =========================
    logger.info("=" * 60)
    logger.info("Stage 2: fine-tuning")
    logger.info("=" * 60)
    info = unfreeze_for_finetune(
        base,
        fine_tune_at=_resolve_fine_tune_at(config["fine_tune_at"], config["backbone"], logger),
        freeze_bn=bool(config["freeze_bn_on_finetune"]),
    )
    logger.info(f"unfreeze info: {info}")
    logger.info(f"params (stage2): {count_params(inner)}")

    if isinstance(model, GradAccumModel):
        model.reset_accumulators()
    _compile(model, build_optimizer(config["optimizer"], float(config["lr_finetune"]), logger), loss, metrics)

    def _make_cbs_stage2():
        out = build_callbacks(
            run_dir / "stage2",
            monitor=monitor,
            es_patience=int(config["early_stopping_patience"]),
            lr_patience=int(config["reduce_lr_patience"]),
            inner_for_ckpt=inner if accum_steps > 1 else None,
        )
        if isinstance(model, GradAccumModel):
            out.append(GradAccumFlushCallback(model))
        return out

    (run_dir / "stage2").mkdir(parents=True, exist_ok=True)
    cbs2 = _make_cbs_stage2()

    hist2, train_ds, val_ds, batch_size = _fit_with_oom_retry(
        model,
        train_ds=train_ds, val_ds=val_ds,
        epochs=int(config["epochs_finetune"]),
        callbacks=cbs2,
        class_weight=class_weight,
        stage_name="Stage2",
        current_batch=batch_size,
        config=config, tp=tp,
        callbacks_builder=_make_cbs_stage2,
        logger=logger,
    )
    log_vram_usage(logger, "after stage2")

    # 4) 최종 저장 (가중치만 저장 → 추론 시 build_model + load_weights)
    final_path = run_dir / "final.weights.h5"
    inner.save_weights(str(final_path))

    # best 체크포인트 검증: stage2 → stage1 → final 순으로 fallback.
    # 셋 다 없으면 학습 자체에 문제가 있었다는 의미이므로 명시적으로 raise.
    candidates = [
        run_dir / "stage2" / "best.weights.h5",
        run_dir / "stage1" / "best.weights.h5",
        final_path,
    ]
    best_path = next((p for p in candidates if p.exists()), None)
    if best_path is None:
        raise FileNotFoundError(
            f"학습 후 저장된 체크포인트를 찾을 수 없음. 확인 경로: "
            f"{[str(p) for p in candidates]}"
        )
    if best_path == final_path:
        logger.warning(
            "best.weights.h5 (stage1/stage2) 가 존재하지 않아 final.weights.h5 로 fallback. "
            "monitor 지표가 개선되지 않았거나 EarlyStopping 이 0 epoch에서 발동했을 수 있음."
        )

    history = {
        "stage1": {k: [float(v) for v in vs] for k, vs in hist1.history.items()},
        "stage2": {k: [float(v) for v in vs] for k, vs in hist2.history.items()},
    }
    save_json(history, str(run_dir / "history.json"))

    return {
        "run_dir": str(run_dir),
        "best_weights": str(best_path),
        "final_weights": str(final_path),
        "history": history,
        "meta": meta,
    }


# ======================================================================
# 데이터셋 빌드 + OOM 폴백 fit
# ======================================================================

def _build_train_datasets(config, tp, batch_size):
    """현재 batch_size로 train/val/test 파이프라인 빌드."""
    return build_datasets(
        data_dir=tp["data_dir"],
        task=tp["task"],
        backbone=config["backbone"],
        img_size=tp["img_size"],
        num_classes=tp["num_classes"],
        batch_size=batch_size,
        use_clahe=bool(config.get("use_clahe", True)),
        shuffle_buffer=int(config.get("shuffle_buffer", 1000)),
        cache=bool(config.get("cache_dataset", False)),
        oversample=bool(config.get("oversample", False)),
        seed=int(config.get("seed", 42)),
    )


def _fit_with_oom_retry(
    model,
    *,
    train_ds,
    val_ds,
    epochs,
    callbacks,
    class_weight,
    stage_name: str,
    current_batch: int,
    config,
    tp,
    callbacks_builder,
    logger,
    max_retries: int = 2,
):
    """model.fit() 호출을 감싸고 tf.errors.ResourceExhaustedError 발생 시
    batch_size를 반감 후 데이터셋·콜백을 재빌드하여 재시도한다.

    Args:
        callbacks_builder: 인자 없이 호출되어 새 콜백 리스트를 반환하는 callable.
            EarlyStopping/ReduceLROnPlateau는 상태를 가지므로 재시도마다 새로 만들어야 함.

    Returns:
        (history, train_ds, val_ds, last_batch_size)
    """
    attempt = 0
    while True:
        try:
            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=2,
            )
            return hist, train_ds, val_ds, current_batch
        except tf.errors.ResourceExhaustedError:
            attempt += 1
            if attempt > max_retries or current_batch <= 1:
                logger.error(
                    f"{stage_name}: OOM 폴백 {max_retries}회 후에도 실패. "
                    f"마지막 batch_size={current_batch}"
                )
                raise
            new_batch = max(1, current_batch // 2)
            logger.warning(
                f"{stage_name}: ResourceExhaustedError → batch_size {current_batch} → {new_batch} 로 재시도 "
                f"(attempt={attempt}/{max_retries})"
            )
            current_batch = new_batch
            # GradAccumModel은 누적 변수 잔여 grad를 비워야 한다.
            if isinstance(model, GradAccumModel):
                model.reset_accumulators()
            train_ds, val_ds, _, _ = _build_train_datasets(config, tp, current_batch)
            callbacks = callbacks_builder()
