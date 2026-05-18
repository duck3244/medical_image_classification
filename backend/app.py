"""
app.py
FastAPI 진입점 — 단일 사용자 MVP.

⚠️ 본 서비스는 개인 학습·연구 목적이며 의료 진단에 사용할 수 없습니다.

실행:
    cd backend
    uvicorn app:app --reload --port 8000

라우트:
    GET  /api/health
    GET  /api/tasks
    GET  /api/models
    POST /api/predict
    GET  /static/results/...   (Grad-CAM 이미지 정적 서빙)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import health, models as models_route, predict, tasks
from services.bootstrap import auto_bootstrap
from services.inference import find_default_weights, warmup
from utils import load_config, setup_gpu_memory_growth


BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "config.json"


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("medimg.api")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(h)
    return logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시: config 로드, GPU memory_growth, 모델 사전 로드(워밍업)."""
    logger = _make_logger()
    logger.info("[Disclaimer] 본 서비스는 학습/연구용이며 의료 진단에 사용할 수 없습니다.")

    cfg = load_config(str(CONFIG_PATH))
    setup_gpu_memory_growth(logger)

    models_dir = BACKEND_DIR / cfg["paths"]["models_dir"]
    app.state.config = cfg
    app.state.config_path = CONFIG_PATH
    app.state.backend_dir = BACKEND_DIR
    app.state.logger = logger

    # 가중치가 없으면 합성 미니 데이터로 자동 부트스트랩 학습 → 결과 가중치를 워밍업.
    weights = find_default_weights(models_dir)
    if weights is None:
        weights = await auto_bootstrap(cfg, BACKEND_DIR, models_dir, logger)

    app.state.default_weights = str(weights) if weights else None

    if weights:
        logger.info(f"default weights: {weights}")
        await warmup(cfg, str(weights))
    else:
        logger.warning(
            "가중치 준비 실패 — predict 호출 시 503을 반환합니다. "
            "수동 학습 또는 weights override가 필요합니다."
        )

    yield
    logger.info("shutdown")


app = FastAPI(
    title="Medical Image Classification API (MVP)",
    description=(
        "단일 사용자 MVP. ⚠️ 본 API는 교육/연구 목적이며 실제 의료 진단/임상 의사결정에 "
        "사용할 수 없습니다. NOT a medical device."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# 개발 시 Vite (5173) ↔ FastAPI (8000) cross-origin 허용.
# 운영 시 frontend dist를 StaticFiles로 같이 서빙하면 동일 출처가 되어 CORS 불필요.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Grad-CAM 결과 이미지를 직접 다운로드/표시할 수 있도록 results/ 를 정적 서빙.
# uploads/ 는 보안상 서빙하지 않는다 (원본 이미지는 응답에 포함하지 않음).
RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(tasks.router, prefix="/api", tags=["tasks"])
app.include_router(models_route.router, prefix="/api", tags=["models"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
