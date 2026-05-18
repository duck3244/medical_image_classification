"""
services/inference.py
TF 모델 로드/캐시 + 단일 GPU 직렬화 락 + predictor 래핑.

설계:
- 단일 사용자 MVP이지만 멀티탭/연속 클릭을 가정해 asyncio.Lock으로 직렬화.
- 모델은 (task, backbone, weights_path) 키로 캐시. weights 교체 시 lazy reload.
- predict_one은 CPU/GPU-bound이므로 asyncio.to_thread로 스레드풀에 위임.
- mixed_precision은 predictor.predict()가 자체적으로 fp32로 강제하지만,
  서비스에서도 진입 시 fp32 정책을 보장한다 (Grad-CAM 안정성).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

from model import build_model
from predictor import predict_one
from utils import get_task_params


logger = logging.getLogger("medimg.api.inference")

# 단일 GPU 보호용 락. FastAPI는 sync def도 스레드풀에서 실행하지만,
# Keras 모델 자체가 thread-safe가 아니라서 명시적 직렬화 필요.
_lock = asyncio.Lock()

# 캐시된 모델 핸들
_model: Optional[tf.keras.Model] = None
_model_key: Optional[Tuple[str, str, str]] = None  # (task, backbone, weights_path)


def _set_fp32_policy_if_mixed() -> None:
    cur = str(tf.keras.mixed_precision.global_policy().name)
    if cur.startswith("mixed"):
        logger.info(f"global policy {cur} → float32 (Grad-CAM 안정성)")
        tf.keras.mixed_precision.set_global_policy("float32")


def _load_model(config: dict, weights_path: str) -> tf.keras.Model:
    """모델 빌드 + 가중치 로드. CPU/GPU 블로킹 작업."""
    tp = get_task_params(config)
    m, _ = build_model(
        backbone=config["backbone"],
        num_classes=tp["num_classes"],
        img_size=tp["img_size"],
    )
    m.load_weights(weights_path)
    return m


async def ensure_model(config: dict, weights_path: str) -> tf.keras.Model:
    """캐시 키가 일치하면 기존 모델 반환, 아니면 새로 빌드/로드."""
    global _model, _model_key
    key = (str(config.get("task")), str(config.get("backbone")), str(weights_path))
    async with _lock:
        if _model is not None and _model_key == key:
            return _model
        _set_fp32_policy_if_mixed()
        logger.info(f"loading model: task={key[0]} backbone={key[1]} weights={key[2]}")
        _model = await asyncio.to_thread(_load_model, config, weights_path)
        _model_key = key
        return _model


async def warmup(config: dict, weights_path: Optional[str]) -> bool:
    """앱 시작 시 사전 로드. weights_path가 None이면 스킵."""
    if not weights_path or not Path(weights_path).exists():
        logger.info("warmup skipped (no preload weights available)")
        return False
    try:
        await ensure_model(config, weights_path)
        logger.info("warmup OK")
        return True
    except Exception as e:
        logger.warning(f"warmup failed: {type(e).__name__}: {e}")
        return False


async def run_predict_one(
    config: dict,
    weights_path: str,
    image_path: str,
    out_dir: Path,
    save_gradcam: bool = True,
    out_stem: Optional[str] = None,
) -> dict:
    """단일 이미지 추론. 모델 로드/교체 → predict_one을 락 안에서 실행."""
    model = await ensure_model(config, weights_path)
    async with _lock:
        return await asyncio.to_thread(
            predict_one,
            config, model, image_path, out_dir, save_gradcam, out_stem,
        )


def find_default_weights(models_dir: Path) -> Optional[Path]:
    """`backend/models/**/stage{2,1}/best.weights.h5` 또는 final.weights.h5 중 최신."""
    candidates = []
    for stage in ("stage2", "stage1"):
        candidates.extend(models_dir.glob(f"*/{stage}/best.weights.h5"))
    candidates.extend(models_dir.glob("*/final.weights.h5"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
