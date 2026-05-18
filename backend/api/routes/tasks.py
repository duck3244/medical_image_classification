"""GET /api/tasks — 지원 가능한 분류 태스크 목록."""
from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/tasks")
def list_tasks(request: Request) -> dict:
    cfg = request.app.state.config
    return {
        "tasks": [
            {
                "id": "nih_cxr_binary",
                "label": "NIH ChestX-ray14 — Pneumonia 이진 분류",
                "classes": ["No_Pneumonia", "Pneumonia"],
                "img_size": cfg.get("img_size_nih", 256),
                "license": "Public Domain",
            },
            {
                "id": "isic_multiclass",
                "label": "ISIC 2019 — 피부병변 7-class",
                "classes": cfg.get(
                    "isic_classes",
                    ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"],
                ),
                "img_size": cfg.get("img_size_isic", 224),
                "license": "CC-BY-NC 4.0 (non-commercial)",
            },
        ],
        "current": cfg.get("task"),
        "available_backbones": cfg.get(
            "available_backbones", ["EfficientNetB0", "MobileNetV2", "ResNet50"]
        ),
    }
