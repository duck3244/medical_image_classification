"""GET /api/models — backend/models 디렉토리를 스캔해 사용 가능한 가중치 목록 반환."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Request


router = APIRouter()


def _scan(models_dir: Path) -> List[dict]:
    """`<task>_<backbone>_<ts>/{stage1,stage2}/best.weights.h5` 와 `final.weights.h5` 수집."""
    out: List[dict] = []
    if not models_dir.exists():
        return out
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        weights = []
        for stage in ("stage1", "stage2"):
            p = run_dir / stage / "best.weights.h5"
            if p.exists():
                weights.append({"stage": stage, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 2)})
        final = run_dir / "final.weights.h5"
        if final.exists():
            weights.append({"stage": "final", "path": str(final), "size_mb": round(final.stat().st_size / 1e6, 2)})
        if not weights:
            continue
        # 디렉토리 이름은 `<task>_<backbone>_<YYYYmmdd_HHMMSS>` 패턴
        meta_path = run_dir / "run_config.json"
        out.append({
            "run_dir": run_dir.name,
            "weights": weights,
            "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
            "has_run_config": meta_path.exists(),
        })
    out.sort(key=lambda m: m["modified"], reverse=True)
    return out


@router.get("/models")
def list_models(request: Request) -> dict:
    models_dir = request.app.state.backend_dir / request.app.state.config["paths"]["models_dir"]
    return {
        "models_dir": str(models_dir),
        "default_weights": request.app.state.default_weights,
        "runs": _scan(models_dir),
    }
