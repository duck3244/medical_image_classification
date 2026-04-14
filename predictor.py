"""
predictor.py
단일/배치 이미지 추론 + Grad-CAM 시각화.

핵심:
- build_model() + load_weights() 로 모델 로드 (custom loss 회피)
- task별 전처리(X-ray CLAHE, ISIC RGB) 일관 적용
- Grad-CAM은 마지막 conv 레이어를 자동 탐색
- 모든 시각화 이미지에 학습용 Disclaimer 워터마크 삽입
- CLI 출력 첫 줄에도 Disclaimer 표시
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from data_processor import _apply_clahe_np
from model import build_model, get_preprocess_fn
from utils import DISCLAIMER_KO, get_task_params, save_json


# ======================================================================
# 입력 전처리
# ======================================================================

def _load_and_prepare(
    image_path: str,
    *,
    task: str,
    backbone: str,
    img_size: int,
    use_clahe: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        display_rgb: 시각화용 uint8 RGB (img_size, img_size, 3)
        model_input: 모델 입력 float32 (1, img_size, img_size, 3) - 백본 preprocess 적용
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if task == "nih_cxr_binary" and use_clahe:
        rgb = _apply_clahe_np(rgb)

    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    preprocess = get_preprocess_fn(backbone)
    x = preprocess(rgb.astype(np.float32))
    x = np.expand_dims(x, axis=0)
    return rgb, x


# ======================================================================
# 모델 로드
# ======================================================================

def load_model_for_inference(config: dict, weights_path: str) -> tf.keras.Model:
    tp = get_task_params(config)
    model, _ = build_model(
        backbone=config["backbone"],
        num_classes=tp["num_classes"],
        img_size=tp["img_size"],
    )
    model.load_weights(weights_path)
    return model


# ======================================================================
# Grad-CAM
# ======================================================================

def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """모델 그래프를 거꾸로 훑어 마지막 Conv2D 레이어 이름 반환.

    reshape/pool 등 비-conv 4D 출력 레이어가 섞여도 오탐하지 않도록
    `tf.keras.layers.Conv2D` 타입을 명시적으로 확인한다.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # 중첩 모델(예: 백본) 내부까지 탐색
        inner_layers = getattr(layer, "layers", None)
        if inner_layers:
            for sub in reversed(inner_layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("Conv2D 레이어를 찾을 수 없습니다.")


def grad_cam(
    model: tf.keras.Model,
    image_input: np.ndarray,
    class_idx: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Returns: (H, W) float32 [0..1] 히트맵
    """
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image_input, training=False)
        if preds.shape[-1] == 1:
            target = preds[:, 0]
        else:
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]).numpy())
            target = preds[:, class_idx]

    grads = tape.gradient(target, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    cam = tf.reduce_sum(conv_out * pooled, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    out = (heatmap * alpha + rgb * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return out


# ======================================================================
# 워터마크
# ======================================================================

def _watermark(img_rgb: np.ndarray, text: str = "교육용 / NOT FOR DIAGNOSIS") -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, img_rgb.shape[0] // 25))
    except Exception:
        font = ImageFont.load_default()
    w, h = pil.size
    bar_h = max(20, h // 18)
    draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0, 160))
    draw.text((8, h - bar_h + 4), text, fill=(255, 255, 255, 255), font=font)
    return np.array(pil)


# ======================================================================
# 단일 추론
# ======================================================================

def predict_one(
    config: dict,
    model: tf.keras.Model,
    image_path: str,
    out_dir: Path,
    save_gradcam: bool = True,
) -> dict:
    tp = get_task_params(config)
    is_binary = tp["is_binary"]

    display_rgb, x = _load_and_prepare(
        image_path,
        task=tp["task"],
        backbone=config["backbone"],
        img_size=tp["img_size"],
        use_clahe=bool(config.get("use_clahe", True)),
    )
    pred = model.predict_on_batch(x)
    pred = np.asarray(pred)[0]

    if is_binary:
        prob = float(pred.reshape(-1)[0])
        pred_idx = int(prob >= 0.5)
        confidence = prob if pred_idx == 1 else 1.0 - prob
        result = {
            "image": str(image_path),
            "predicted_class": tp["class_names"][pred_idx],
            "predicted_index": pred_idx,
            "probability_positive": prob,
            "confidence": float(confidence),
        }
    else:
        probs = pred.astype(float)
        pred_idx = int(np.argmax(probs))
        result = {
            "image": str(image_path),
            "predicted_class": tp["class_names"][pred_idx],
            "predicted_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {
                name: float(p) for name, p in zip(tp["class_names"], probs)
            },
        }

    if save_gradcam:
        try:
            cam = grad_cam(model, x, class_idx=pred_idx if not is_binary else None)
            overlay = overlay_heatmap(display_rgb, cam)
            stamped = _watermark(
                overlay,
                f"{result['predicted_class']} ({result['confidence']*100:.1f}%) "
                f"— 교육용 / NOT FOR DIAGNOSIS",
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(image_path).stem}_gradcam.png"
            Image.fromarray(stamped).save(out_path)
            result["gradcam_path"] = str(out_path)
        except (ImportError, ValueError, RuntimeError, tf.errors.InvalidArgumentError) as e:
            import warnings
            warnings.warn(f"Grad-CAM 생성 실패: {type(e).__name__}: {e}")
            result["gradcam_error"] = f"{type(e).__name__}: {e}"

    return result


# ======================================================================
# 배치 추론 (디렉토리)
# ======================================================================

def predict_dir(
    config: dict,
    model: tf.keras.Model,
    input_dir: str,
    out_dir: Path,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> List[dict]:
    paths = [
        p for p in sorted(Path(input_dir).rglob("*"))
        if p.suffix.lower() in exts
    ]
    results = []
    for p in paths:
        results.append(predict_one(config, model, str(p), out_dir))
    return results


# ======================================================================
# 진입점
# ======================================================================

def predict(
    config: dict,
    weights_path: str,
    image: Optional[str],
    input_dir: Optional[str],
    logger,
) -> dict:
    print(">>> [DISCLAIMER] 본 결과는 교육·학습용입니다. 진단에 사용할 수 없습니다.")
    model = load_model_for_inference(config, weights_path)
    logger.info(f"weights loaded: {weights_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config["paths"]["results_dir"]) / f"predict_{ts}"

    if image:
        result = predict_one(config, model, image, out_dir)
        save_json(result, str(out_dir / "result.json"))
        logger.info(f"prediction: {result}")
        return {"results": [result], "out_dir": str(out_dir)}

    if input_dir:
        results = predict_dir(config, model, input_dir, out_dir)
        save_json({"results": results}, str(out_dir / "results.json"))
        logger.info(f"{len(results)} images processed → {out_dir}")
        return {"results": results, "out_dir": str(out_dir)}

    raise ValueError("--image 또는 --input_dir 중 하나는 필요합니다.")
