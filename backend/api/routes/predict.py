"""POST /api/predict — 단일 이미지 업로드 추론.

업로드 파일 → backend/uploads/{uuid}.{ext} 로 저장 → predictor 실행 →
Grad-CAM 결과는 backend/results/predict_{ts}/{stem}_gradcam.png 로 저장됨.

응답은 예측 결과 JSON과 Grad-CAM 다운로드 URL을 포함. 모든 응답에 disclaimer 포함.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from services.inference import run_predict_one
from utils import get_task_params


router = APIRouter()


_ALLOWED_EXT = {".png", ".jpg", ".jpeg"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

_DISCLAIMER = (
    "본 결과는 교육·연구 목적이며 실제 의료 진단/임상 의사결정에 사용할 수 없습니다. "
    "NOT a medical device. NOT for diagnosis."
)


def _validate_upload(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 확장자: {ext or '(없음)'}. 허용: {sorted(_ALLOWED_EXT)}",
        )
    return ext


async def _save_upload(file: UploadFile, dst_dir: Path, ext: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{uuid.uuid4().hex}{ext}"
    size = 0
    with dst.open("wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                dst.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"파일 크기 초과: {_MAX_UPLOAD_BYTES // (1024*1024)} MB 이내여야 함",
                )
            f.write(chunk)
    return dst


def _build_request_config(app_cfg: dict, task: Optional[str], backbone: Optional[str]) -> dict:
    """task/backbone override를 반영한 임시 config."""
    cfg = dict(app_cfg)
    if task:
        cfg["task"] = task
    if backbone:
        cfg["backbone"] = backbone
    # task 변경 시 num_classes/img_size 자동 결정은 get_task_params가 처리.
    return cfg


@router.post("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    task: Optional[str] = Form(default=None),
    backbone: Optional[str] = Form(default=None),
    weights: Optional[str] = Form(default=None),
):
    """단일 이미지 추론 + Grad-CAM 생성.

    Form fields:
        file:     이미지 파일 (png/jpg, 10 MB 이하)
        task:     (선택) 'nih_cxr_binary' | 'isic_multiclass' — config 덮어쓰기
        backbone: (선택) 'EfficientNetB0' | 'MobileNetV2' | 'ResNet50' — config 덮어쓰기
        weights:  (선택) 사용할 가중치 절대 경로. 미지정 시 lifespan에서 결정한 default_weights 사용.
    """
    ext = _validate_upload(file)

    app_cfg = request.app.state.config
    cfg = _build_request_config(app_cfg, task, backbone)

    weights_path = weights or request.app.state.default_weights
    if not weights_path:
        raise HTTPException(
            status_code=503,
            detail=(
                "사용 가능한 가중치가 없습니다. backend/models/ 에 학습된 weights가 있어야 하며, "
                "없다면 먼저 `python main.py --mode train ...` 으로 학습을 수행하세요."
            ),
        )

    # task가 변경되었으면 num_classes 등을 검증
    try:
        get_task_params(cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    backend_dir: Path = request.app.state.backend_dir
    upload_dir = backend_dir / "uploads"
    src_path = await _save_upload(file, upload_dir, ext)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = backend_dir / cfg["paths"]["results_dir"] / f"predict_{ts}_{uuid.uuid4().hex[:6]}"

    try:
        result = await run_predict_one(
            cfg,
            weights_path=weights_path,
            image_path=str(src_path),
            out_dir=out_dir,
            save_gradcam=True,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    # results/ 는 /static/results/ 로 마운트되어 있음
    gradcam_url = None
    if "gradcam_path" in result:
        rel = Path(result["gradcam_path"]).resolve().relative_to(
            (backend_dir / "results").resolve()
        )
        gradcam_url = f"/static/results/{rel.as_posix()}"

    return {
        "disclaimer": _DISCLAIMER,
        "task": cfg["task"],
        "backbone": cfg["backbone"],
        "weights": weights_path,
        "uploaded_filename": file.filename,
        "result": {
            "predicted_class": result.get("predicted_class"),
            "predicted_index": result.get("predicted_index"),
            "confidence": result.get("confidence"),
            "probability_positive": result.get("probability_positive"),
            "class_probabilities": result.get("class_probabilities"),
            "gradcam_url": gradcam_url,
            "gradcam_error": result.get("gradcam_error"),
        },
        "run_dir": str(out_dir.relative_to(backend_dir)),
    }
