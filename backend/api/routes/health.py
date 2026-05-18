"""GET /api/health — 서버/GPU/모델 상태 점검."""
from __future__ import annotations

from fastapi import APIRouter, Request
import tensorflow as tf

# 모듈을 임포트해서 _model_key를 매 요청마다 동적으로 읽음
# (값을 직접 임포트하면 임포트 시점 None이 고정됨)
from services import inference


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    cfg = request.app.state.config
    return {
        "status": "ok",
        "disclaimer": (
            "본 서비스는 교육·연구 목적이며 실제 의료 진단에 사용할 수 없습니다. "
            "NOT a medical device."
        ),
        "tensorflow": tf.__version__,
        "gpu_count": len(gpus),
        "gpu_names": [g.name for g in gpus],
        "current_task": cfg.get("task"),
        "current_backbone": cfg.get("backbone"),
        "loaded_model_key": inference._model_key,
        "default_weights": request.app.state.default_weights,
    }
