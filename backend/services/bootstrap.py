"""
services/bootstrap.py
서버 시작 시 학습 가중치가 없으면 합성 미니 데이터셋으로 빠른 학습을 수행하여
즉시 사용 가능한 데모 모델을 만든다.

⚠️ 부트스트랩 가중치는 인공 노이즈 + 단순 시각 신호(우폐 패치)로 학습된 것으로
   임상적 의미가 전혀 없으며, UI/파이프라인 동작 검증 용도로만 사용해야 한다.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


# 부트스트랩이 생성하는 데이터셋 디렉토리명. dataset/<이 이름> 하위에 저장.
BOOTSTRAP_DATA_NAME = "nih_cxr_bootstrap"


def generate_synthetic_dataset(data_dir: Path, logger: logging.Logger) -> None:
    """nih_cxr_binary 매니페스트 형식의 합성 미니 데이터셋 생성.

    train/val/test = 40/10/10 = 60장. 양성률 40%. 양성에는 우폐 중앙에
    밝은 패치를 삽입해 모델이 시각적으로 분리 가능한 신호를 학습하도록 한다.
    """
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def make_cxr(seed: int, positive: bool) -> Image.Image:
        rng = np.random.default_rng(seed)
        base = rng.normal(120, 30, size=(256, 256)).clip(0, 255)
        y, x = np.ogrid[:256, :256]
        # 폐 영역 시뮬: 양쪽에 어두운 타원
        for cx in (90, 166):
            mask = ((x - cx) ** 2 / 60**2 + (y - 130) ** 2 / 90**2) < 1
            base[mask] *= 0.6
        # 양성 시그널: 우폐 중앙에 밝은 consolidation
        if positive:
            cx = 160 + int(rng.integers(-15, 15))
            cy = 140 + int(rng.integers(-15, 15))
            mask = ((x - cx) ** 2 + (y - cy) ** 2) < (20 + int(rng.integers(0, 8))) ** 2
            base[mask] = np.clip(base[mask] + rng.uniform(60, 100), 0, 255)
        rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)
        return Image.fromarray(rgb)

    seed_base = {"train": 0, "val": 1000, "test": 2000}
    n_split = {"train": 40, "val": 10, "test": 10}
    pos_ratio = 0.4
    random.seed(7)

    for split, n in n_split.items():
        rows = []
        for i in range(n):
            positive = i < int(round(n * pos_ratio))
            seed = seed_base[split] + i
            img = make_cxr(seed, positive)
            rel = img_dir / f"{split}_{i:03d}.png"
            img.save(rel)
            rows.append({"filepath": str(rel), "label": 1 if positive else 0})
        random.shuffle(rows)
        df = pd.DataFrame(rows)
        df.to_csv(data_dir / f"{split}.csv", index=False)
        pos = int((df["label"] == 1).sum())
        logger.info(f"  bootstrap {split}: {len(df)} (pos={pos})")


def run_bootstrap_training(
    config: dict, backend_dir: Path, logger: logging.Logger
) -> Optional[Path]:
    """합성 데이터로 1+1 epoch 빠른 학습 수행. best.weights.h5 절대 경로 반환.

    config 의 backbone / img_size_nih 는 그대로 사용하여 lifespan 워밍업과
    모델 형상이 일치하도록 한다.
    """
    data_dir = backend_dir / "dataset" / BOOTSTRAP_DATA_NAME
    if not (data_dir / "train.csv").exists():
        logger.info(f"[bootstrap] synthetic dataset 생성: {data_dir}")
        generate_synthetic_dataset(data_dir, logger)
    else:
        logger.info(f"[bootstrap] 기존 synthetic dataset 재사용: {data_dir}")

    # 부트스트랩 전용 임시 config — backbone/img_size 는 유지하고 paths/epochs만 교체.
    boot_cfg = dict(config)
    boot_cfg["paths"] = dict(config["paths"])
    boot_cfg["paths"]["nih_dir"] = str(
        (data_dir).relative_to(backend_dir)
    )
    boot_cfg["task"] = "nih_cxr_binary"
    boot_cfg["epochs_head"] = 1
    boot_cfg["epochs_finetune"] = 1
    boot_cfg["batch_size"] = min(8, int(config.get("batch_size", 8)))
    boot_cfg["grad_accum_steps"] = 1
    boot_cfg["mixed_precision"] = False  # Grad-CAM 안정성 + 작은 모델이라 의미 없음
    boot_cfg["xla_jit"] = False
    boot_cfg["shuffle_buffer"] = 200
    boot_cfg["oversample"] = True
    boot_cfg["focal_alpha_auto"] = True

    logger.info("=" * 60)
    logger.info("[bootstrap] 합성 데이터로 1+1 epoch 빠른 학습 시작")
    logger.info("⚠️  결과 가중치는 데모용이며 임상적 의미 없음 (NOT for diagnosis)")
    logger.info("=" * 60)

    # trainer 는 paths.*가 상대경로일 때 cwd 기준으로 해석하므로 backend/ 로 chdir.
    cwd_orig = os.getcwd()
    try:
        os.chdir(backend_dir)
        from trainer import train as train_fn  # 지연 import (TF 무거움)

        result = train_fn(boot_cfg, logger)
        best = Path(result["best_weights"])
        if not best.is_absolute():
            best = (backend_dir / best).resolve()
        logger.info(f"[bootstrap] 학습 완료: {best}")
        return best
    finally:
        os.chdir(cwd_orig)


async def auto_bootstrap(
    config: dict,
    backend_dir: Path,
    models_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """가중치가 없으면 비동기로 부트스트랩 학습 실행. 기존 가중치가 있으면 그대로 반환.

    학습은 blocking 작업이므로 asyncio.to_thread 로 디스패치한다. 호출 시점은
    lifespan startup이라 동시 요청이 없어 안전.
    """
    from .inference import find_default_weights

    existing = find_default_weights(models_dir)
    if existing is not None:
        return existing

    logger.info("[bootstrap] 학습된 가중치 없음 → 자동 부트스트랩 시작")
    try:
        path = await asyncio.to_thread(
            run_bootstrap_training, config, backend_dir, logger
        )
        return path
    except Exception as e:  # noqa: BLE001 — 부트스트랩 실패는 서버 시작을 막지 않음
        logger.exception(
            f"[bootstrap] 자동 학습 실패: {type(e).__name__}: {e}. "
            "서버는 가중치 없이 시작합니다 — predict는 503을 반환합니다."
        )
        return None
