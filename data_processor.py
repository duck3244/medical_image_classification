"""
data_processor.py
tf.data 기반 학습/검증/테스트 파이프라인.

설계 포인트:
- 매니페스트(CSV: filepath,label) 기반으로 일관된 인터페이스
- X-ray: grayscale → CLAHE → 3채널 복제 (CLAHE는 CPU에서 tf.py_function)
- 피부병변: RGB 로드 → 리사이즈 → 색상 증강 최소화
- 증강은 학습 split에만 적용
- 백본별 preprocess_input을 마지막 단계에 적용
- class_weight는 sklearn으로 자동 산출
- RTX 4060 8GB 기준 prefetch/parallel 보수적 설정
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from model import get_preprocess_fn


AUTOTUNE = tf.data.AUTOTUNE

# tensorflow_addons는 선택적 의존성. 부재 시 회전 증강은 비활성화되며 최초 1회 경고한다.
try:
    import tensorflow_addons as _tfa  # type: ignore
    _TFA_AVAILABLE = True
except ImportError:
    _tfa = None
    _TFA_AVAILABLE = False
    import warnings
    warnings.warn(
        "tensorflow_addons 미설치 — X-ray 회전 증강이 비활성화됩니다. "
        "설치: pip install tensorflow-addons",
        RuntimeWarning,
        stacklevel=2,
    )


# ======================================================================
# 매니페스트 로드
# ======================================================================

def load_manifest(csv_path: str) -> pd.DataFrame:
    """
    CSV 매니페스트 로드. 필수 컬럼: filepath, label
    - filepath: 이미지 절대/상대 경로
    - label: 정수 (0..num_classes-1)
    """
    df = pd.read_csv(csv_path)
    required = {"filepath", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    df["label"] = df["label"].astype(np.int32)
    return df


def compute_class_weights(labels: np.ndarray) -> dict:
    """balanced class weight (dict[int, float])."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ======================================================================
# 이미지 로드 & 전처리
# ======================================================================

def _decode_image(path: tf.Tensor, channels: int = 3) -> tf.Tensor:
    """JPG/PNG 공용 디코더. uint8 반환."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=channels, expand_animations=False)
    img.set_shape([None, None, channels])
    return img


def _resize(img: tf.Tensor, size: int) -> tf.Tensor:
    return tf.image.resize(img, [size, size], method=tf.image.ResizeMethod.BILINEAR)


def _apply_clahe_np(img_uint8: np.ndarray) -> np.ndarray:
    """
    CPU에서 OpenCV CLAHE 수행. 입력은 uint8 RGB(H,W,3) 또는 grayscale(H,W).
    출력: uint8 RGB(H,W,3) - X-ray용으로 grayscale→CLAHE→3ch 복제.
    """
    import cv2
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8.squeeze()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray.astype(np.uint8))
    rgb = np.stack([eq, eq, eq], axis=-1)
    return rgb.astype(np.uint8)


def _tf_clahe(img: tf.Tensor) -> tf.Tensor:
    """tf.py_function 래퍼. uint8 in → uint8 out."""
    def _fn(x):
        return _apply_clahe_np(x.numpy())
    out = tf.py_function(_fn, [img], Tout=tf.uint8)
    out.set_shape([None, None, 3])
    return out


# ======================================================================
# 증강 (task별)
# ======================================================================

def _augment_xray(img: tf.Tensor) -> tf.Tensor:
    """
    흉부 X-ray 증강 (강화): horizontal flip은 여전히 금지 (좌우 해부학 보존).
    - brightness/contrast 범위 확대
    - random zoom-in crop (최대 10%)
    - small random translation (~8%)
    - 작은 회전 (±6도) via tf.keras preprocessing
    """
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

    # Random zoom-in crop: 90~100% 영역을 crop 후 원 크기로 resize
    shape = tf.shape(img)
    h = shape[0]; w = shape[1]
    scale = tf.random.uniform([], 0.90, 1.00)
    ch = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    offset_h = tf.random.uniform([], 0, h - ch + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, w - cw + 1, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, ch, cw)
    img = tf.image.resize(img, [h, w], method=tf.image.ResizeMethod.BILINEAR)

    # 작은 회전: ±6도 (라디안 0.105). tfa가 있을 때만 적용.
    if _TFA_AVAILABLE:
        angle = tf.random.uniform([], -0.105, 0.105)
        img = _tfa.image.rotate(img, angle, interpolation="BILINEAR")
    return img


def _augment_skin(img: tf.Tensor) -> tf.Tensor:
    """
    피부병변 증강: 회전 무관, flip 양방향 허용, 약한 color jitter.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    return img


# ======================================================================
# 파이프라인 빌더
# ======================================================================

def build_pipeline(
    manifest: pd.DataFrame,
    *,
    task: str,
    backbone: str,
    img_size: int,
    num_classes: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    use_clahe: bool,
    shuffle_buffer: int = 1000,
    cache: bool = False,
    oversample: bool = False,
) -> tf.data.Dataset:
    """
    Args:
        task: "nih_cxr_binary" | "isic_multiclass"
        backbone: 백본 이름 (preprocess_input 선택용)
    """
    preprocess_input = get_preprocess_fn(backbone)
    is_binary = num_classes == 2

    def _build_sample_ds(sub_manifest: pd.DataFrame) -> tf.data.Dataset:
        paths = sub_manifest["filepath"].astype(str).values
        labels = sub_manifest["label"].astype(np.int32).values
        d = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            d = d.shuffle(
                buffer_size=min(shuffle_buffer, max(1, len(sub_manifest))),
                reshuffle_each_iteration=True,
                seed=42,
            ).repeat() if oversample else d.shuffle(
                buffer_size=min(shuffle_buffer, max(1, len(sub_manifest))),
                reshuffle_each_iteration=True,
                seed=42,
            )

        def _load(path, label):
            img = _decode_image(path, channels=3)
            img = tf.cast(img, tf.uint8)
            return img, label
        d = d.map(_load, num_parallel_calls=AUTOTUNE)

        if task == "nih_cxr_binary" and use_clahe:
            def _clahe_map(img, label):
                return _tf_clahe(img), label
            d = d.map(_clahe_map, num_parallel_calls=AUTOTUNE)

        def _resize_map(img, label):
            img = tf.cast(img, tf.float32)
            img = _resize(img, img_size)
            return img, label
        d = d.map(_resize_map, num_parallel_calls=AUTOTUNE)

        if augment:
            aug_fn = _augment_xray if task == "nih_cxr_binary" else _augment_skin
            def _aug_map(img, label):
                return aug_fn(img), label
            d = d.map(_aug_map, num_parallel_calls=AUTOTUNE)

        def _finalize(img, label):
            img = preprocess_input(img)
            if is_binary:
                y = tf.cast(label, tf.float32)
                y = tf.reshape(y, [1])
            else:
                y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
            return img, y
        d = d.map(_finalize, num_parallel_calls=AUTOTUNE)
        return d

    if oversample and is_binary and shuffle:
        # 양성/음성 각각 repeat하여 50:50으로 샘플링 (binary only).
        pos_df = manifest[manifest["label"] == 1]
        neg_df = manifest[manifest["label"] == 0]
        if len(pos_df) == 0 or len(neg_df) == 0:
            ds = _build_sample_ds(manifest)
        else:
            ds_pos = _build_sample_ds(pos_df)
            ds_neg = _build_sample_ds(neg_df)
            ds = tf.data.Dataset.sample_from_datasets(
                [ds_neg, ds_pos], weights=[0.5, 0.5], seed=42
            )
            # 1 epoch 길이를 원 manifest 크기로 한정
            steps_per_epoch = len(manifest)
            ds = ds.take(steps_per_epoch)
    else:
        ds = _build_sample_ds(manifest)
    # 학습 split은 drop_remainder=True로 binary label reshape([1])과의 shape 일관성 확보.
    # val/test는 모든 샘플 평가를 위해 유지.
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ======================================================================
# 편의 함수: 3분할 매니페스트 → train/val/test 파이프라인
# ======================================================================

def build_datasets(
    data_dir: str,
    *,
    task: str,
    backbone: str,
    img_size: int,
    num_classes: int,
    batch_size: int,
    use_clahe: bool,
    shuffle_buffer: int = 1000,
    cache: bool = False,
    oversample: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    data_dir 하위에 train.csv / val.csv / test.csv 매니페스트가 있어야 한다.
    Returns: (train_ds, val_ds, test_ds, meta)
        meta: {"class_weights": dict, "sizes": {...}}
    """
    data_dir = Path(data_dir)
    train_df = load_manifest(data_dir / "train.csv")
    val_df = load_manifest(data_dir / "val.csv")
    test_df = load_manifest(data_dir / "test.csv")

    common = dict(
        task=task,
        backbone=backbone,
        img_size=img_size,
        num_classes=num_classes,
        batch_size=batch_size,
        use_clahe=use_clahe,
        shuffle_buffer=shuffle_buffer,
    )

    train_ds = build_pipeline(
        train_df, shuffle=True, augment=True, cache=False,
        oversample=oversample, **common
    )
    val_ds = build_pipeline(val_df, shuffle=False, augment=False, cache=cache, **common)
    test_ds = build_pipeline(test_df, shuffle=False, augment=False, cache=cache, **common)

    meta = {
        "class_weights": compute_class_weights(train_df["label"].values),
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    return train_ds, val_ds, test_ds, meta


# ======================================================================
# Smoke test (더미 이미지 10장)
# ======================================================================

if __name__ == "__main__":
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        for i in range(10):
            p = tmp / f"img_{i}.png"
            arr = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(p)
            rows.append({"filepath": str(p), "label": i % 2})
        df = pd.DataFrame(rows)

        ds = build_pipeline(
            df,
            task="nih_cxr_binary",
            backbone="EfficientNetB0",
            img_size=256,
            num_classes=2,
            batch_size=4,
            shuffle=True,
            augment=True,
            use_clahe=True,
        )
        for x, y in ds.take(1):
            print("X:", x.shape, x.dtype, "Y:", y.shape, y.dtype)

        print("class_weights:", compute_class_weights(df["label"].values))
