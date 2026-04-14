"""
dataset_downloader.py
NIH ChestX-ray14 / ISIC 2019 데이터셋 다운로드 및 매니페스트 생성.

라이선스:
- NIH ChestX-ray14: Public Domain (U.S. NIH Clinical Center)
- ISIC 2019: CC-BY-NC 4.0 (비상업적 연구 목적만)

두 데이터셋 모두 --agree 플래그로 명시적 라이선스 동의가 필요.
출력: <data_dir>/train.csv, val.csv, test.csv (filepath,label)

주의:
- NIH 전체 이미지 tar.gz은 ~42GB. --subset 0.1 로 일부만 받을 수 있음.
- NIH Box 직링크는 공식 페이지(https://nihcc.app.box.com/v/ChestXray-NIHCC)
  의 링크와 동일. 링크가 만료되는 경우 README를 참고하여 수동 배치.
- 네트워크 이슈 시 수동 다운로드 후 raw/ 하위에 배치하면 매니페스트만 재생성 가능.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from utils import get_logger, print_disclaimer, ensure_dirs


# ======================================================================
# 라이선스 고지
# ======================================================================

NIH_LICENSE = """
[NIH ChestX-ray14 License]
Source : U.S. National Institutes of Health Clinical Center
URL    : https://nihcc.app.box.com/v/ChestXray-NIHCC
License: Public Domain (U.S. Government work)
Use    : Research / educational use permitted.
Citation: Wang X. et al., CVPR 2017.
"""

ISIC_LICENSE = """
[ISIC 2019 Challenge Dataset License]
Source : International Skin Imaging Collaboration
URL    : https://challenge.isic-archive.com/data/#2019
License: CC-BY-NC 4.0 (Non-Commercial)
Use    : Non-commercial research / educational use only.
Citation: Tschandl 2018, Codella 2019, Combalia 2019.
"""


def show_license_and_require_agree(text: str, agreed: bool, logger):
    print(text)
    if not agreed:
        logger.error(
            "라이선스 동의가 필요합니다. `--agree` 플래그를 추가해서 재실행하세요."
        )
        sys.exit(2)
    logger.info("라이선스 동의 확인됨 (--agree)")


# ======================================================================
# NIH ChestX-ray14
# ======================================================================

# 공식 Box 직링크 (NIH Clinical Center 공개)
NIH_IMAGE_URLS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]

NIH_METADATA_URLS = {
    "Data_Entry_2017.csv": "https://nihcc.box.com/shared/static/7jmomz50glzbq7drvtnh4zi9r6ywwkp6.csv",
    "train_val_list.txt": "https://nihcc.box.com/shared/static/i7pjiyppv06qhfyyh1m3shlh41yrxh59.txt",
    "test_list.txt": "https://nihcc.box.com/shared/static/tl5jq5b0tgy8u54vte8mgfs8pqaakl8l.txt",
}


def _download(url: str, dst: Path, logger) -> bool:
    """Resumable 다운로드. 이미 존재하면 skip."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        logger.info(f"skip (exists): {dst.name}")
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dst.name, ncols=80
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        tmp.rename(dst)
        return True
    except Exception as e:
        logger.error(f"다운로드 실패 {url}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def _extract_tar(tar_path: Path, dst_dir: Path, logger):
    logger.info(f"extracting {tar_path.name} → {dst_dir}")
    with tarfile.open(tar_path, "r:gz") as tf_:
        tf_.extractall(dst_dir)


def download_nih(
    nih_dir: str,
    subset: float = 1.0,
    keep_archives: bool = False,
    agreed: bool = False,
    source: str = "huggingface",
    logger=None,
) -> bool:
    """
    Args:
        source: "huggingface" (권장, 안정) | "box" (레거시, 만료 가능) | "manual" (이미 배치된 파일만 매니페스트)
        subset: 0 < subset <= 1.0 — NIH 이미지 archive 12개 중 앞부분만 사용.
    """
    if logger is None:
        logger = get_logger("downloader")
    show_license_and_require_agree(NIH_LICENSE, agreed, logger)

    if source == "huggingface":
        return _download_nih_hf(nih_dir, subset, logger)
    if source == "manual":
        return _build_manifest_only_nih(nih_dir, logger)
    if source == "box":
        return _download_nih_box(nih_dir, subset, keep_archives, logger)
    raise ValueError(f"Unknown source: {source}")


def _download_nih_hf(nih_dir: str, subset: float, logger) -> bool:
    """HuggingFace 미러(alkzar90/NIH-Chest-X-ray-dataset) 사용."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub 미설치. `pip install huggingface_hub` 후 재시도.")
        return False

    nih_dir = Path(nih_dir)
    img_dir = nih_dir / "images"
    meta_dir = nih_dir / "meta"
    ensure_dirs(img_dir, meta_dir)

    repo = "alkzar90/NIH-Chest-X-ray-dataset"

    # 1) 메타데이터 - HF 미러는 v2020 CSV이며 컬럼 동일
    logger.info("HF: 메타데이터 다운로드")
    csv_path = hf_hub_download(repo, "data/Data_Entry_2017_v2020.csv", repo_type="dataset")
    train_val_path = hf_hub_download(repo, "data/train_val_list.txt", repo_type="dataset")
    test_path = hf_hub_download(repo, "data/test_list.txt", repo_type="dataset")
    shutil.copy(csv_path, meta_dir / "Data_Entry_2017.csv")
    shutil.copy(train_val_path, meta_dir / "train_val_list.txt")
    shutil.copy(test_path, meta_dir / "test_list.txt")

    # 2) 이미지 zip (subset만큼)
    total = 12
    n_files = max(1, int(round(total * subset)))
    logger.info(f"HF: image archives {n_files}/{total} (subset={subset})")
    for i in range(1, n_files + 1):
        fname = f"data/images/images_{i:03d}.zip"
        logger.info(f"HF: downloading {fname}")
        local = hf_hub_download(repo, fname, repo_type="dataset")
        logger.info(f"HF: extracting {fname}")
        with zipfile.ZipFile(local, "r") as zf:
            zf.extractall(img_dir)
        # zip 내부 'images/' 폴더로 한 번 감싸짐 → 평탄화
        inner = img_dir / "images"
        if inner.exists():
            for p in inner.glob("*.png"):
                target = img_dir / p.name
                if not target.exists():
                    shutil.move(str(p), target)
            shutil.rmtree(inner, ignore_errors=True)

    build_nih_manifest(nih_dir, logger=logger)
    logger.info("NIH (HF) 다운로드 완료")
    return True


def _build_manifest_only_nih(nih_dir: str, logger) -> bool:
    """이미 배치된 이미지/메타데이터에 대해 매니페스트만 생성."""
    nih_dir = Path(nih_dir)
    expected = [
        nih_dir / "meta" / "Data_Entry_2017.csv",
        nih_dir / "meta" / "train_val_list.txt",
        nih_dir / "meta" / "test_list.txt",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        logger.error(
            "수동 모드 필수 파일 누락:\n  " + "\n  ".join(missing) +
            "\nNIH 공식 페이지(https://nihcc.app.box.com/v/ChestXray-NIHCC)에서 받아 배치하세요."
        )
        return False
    if not any((nih_dir / "images").rglob("*.png")):
        logger.error(f"이미지가 없습니다: {nih_dir / 'images'}/*.png")
        return False
    build_nih_manifest(nih_dir, logger=logger)
    logger.info("NIH (manual) 매니페스트 생성 완료")
    return True


def _download_nih_box(nih_dir: str, subset: float, keep_archives: bool, logger) -> bool:
    """레거시 Box 직링크 (URL이 만료될 수 있음)."""
    nih_dir = Path(nih_dir)
    raw_dir = nih_dir / "raw"
    img_dir = nih_dir / "images"
    meta_dir = nih_dir / "meta"
    ensure_dirs(raw_dir, img_dir, meta_dir)

    for name, url in NIH_METADATA_URLS.items():
        if not _download(url, meta_dir / name, logger):
            return False

    n_files = max(1, int(round(len(NIH_IMAGE_URLS) * subset)))
    logger.info(f"Box: image archives {n_files}/{len(NIH_IMAGE_URLS)} (subset={subset})")
    for i, url in enumerate(NIH_IMAGE_URLS[:n_files]):
        dst = raw_dir / f"images_{i+1:02d}.tar.gz"
        if not _download(url, dst, logger):
            return False
        _extract_tar(dst, img_dir, logger)
        if not keep_archives:
            dst.unlink(missing_ok=True)

    build_nih_manifest(nih_dir, logger=logger)
    logger.info("NIH (Box) 다운로드 완료")
    return True


def build_nih_manifest(nih_dir: str, logger=None):
    """
    Data_Entry_2017.csv + 공식 split 파일을 기반으로 train/val/test 매니페스트 생성.
    라벨: 'Pneumonia' 포함 여부 (이진).
    """
    if logger is None:
        logger = get_logger("downloader")
    nih_dir = Path(nih_dir)
    meta = nih_dir / "meta"
    entry_csv = meta / "Data_Entry_2017.csv"
    train_val_list = meta / "train_val_list.txt"
    test_list = meta / "test_list.txt"

    for p in [entry_csv, train_val_list, test_list]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    df = pd.read_csv(entry_csv)
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "labels"})
    df["label"] = df["labels"].apply(
        lambda s: 1 if "Pneumonia" in str(s).split("|") else 0
    )

    # 실제로 존재하는 이미지만 필터 (subset 다운로드 대응)
    img_dir = nih_dir / "images"
    existing = {p.name for p in img_dir.rglob("*.png")}
    df = df[df["image"].isin(existing)].copy()
    df["filepath"] = df["image"].apply(lambda n: str(img_dir / n))

    train_val = set(train_val_list.read_text().split())
    test_set = set(test_list.read_text().split())

    train_val_df = df[df["image"].isin(train_val)].copy()
    test_df = df[df["image"].isin(test_set)].copy()

    # train/val stratified 10% split
    from sklearn.model_selection import train_test_split
    if len(train_val_df) > 0:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.1,
            stratify=train_val_df["label"],
            random_state=42,
        )
    else:
        train_df = train_val_df
        val_df = train_val_df.iloc[0:0]

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = d[["filepath", "label"]].reset_index(drop=True)
        out.to_csv(nih_dir / f"{name}.csv", index=False)
        pos = int((out["label"] == 1).sum())
        logger.info(f"NIH {name}: {len(out)} (Pneumonia={pos})")


# ======================================================================
# ISIC 2019
# ======================================================================

ISIC_URLS = {
    "ISIC_2019_Training_Input.zip":
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
    "ISIC_2019_Training_GroundTruth.csv":
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
    # lesion_id 기반 patient/lesion-level split을 위한 메타데이터.
    "ISIC_2019_Training_Metadata.csv":
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
}

ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]  # UNK 제외


def download_isic(
    isic_dir: str,
    agreed: bool = False,
    logger=None,
) -> bool:
    if logger is None:
        logger = get_logger("downloader")
    show_license_and_require_agree(ISIC_LICENSE, agreed, logger)

    isic_dir = Path(isic_dir)
    raw_dir = isic_dir / "raw"
    img_dir = isic_dir / "images"
    ensure_dirs(raw_dir, img_dir)

    # 1) 다운로드
    for name, url in ISIC_URLS.items():
        if not _download(url, raw_dir / name, logger):
            return False

    # 2) 압축 해제
    zip_path = raw_dir / "ISIC_2019_Training_Input.zip"
    if not any(img_dir.glob("*.jpg")):
        logger.info(f"extracting {zip_path.name} → {img_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(img_dir)
        # ZIP 내부가 폴더로 한 번 더 감쌀 수 있음 → 평탄화
        inner = img_dir / "ISIC_2019_Training_Input"
        if inner.exists():
            for p in inner.glob("*.jpg"):
                shutil.move(str(p), img_dir / p.name)
            shutil.rmtree(inner, ignore_errors=True)

    # 3) 매니페스트 생성
    build_isic_manifest(isic_dir, logger=logger)
    logger.info("ISIC 다운로드 완료")
    return True


def _isic_lesion_group_split(df: pd.DataFrame, logger):
    """
    StratifiedGroupKFold를 활용한 lesion-level split.
    동일 lesion_id를 가진 모든 이미지는 하나의 split(train/val/test)에만 배치되어
    데이터 누수를 방지한다. 클래스 분포는 가능한 한 보존.

    Returns: (train_df, val_df, test_df)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    # test ≈ 10%, val ≈ 10%, train ≈ 80% 가 되도록 10-fold에서 1 fold씩 빼낸다.
    skf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, test_idx = next(
        skf_test.split(df, df["label"], groups=df["lesion_id"])
    )
    test_df = df.iloc[test_idx].copy()
    rest_df = df.iloc[train_idx].copy().reset_index(drop=True)

    skf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
    train_idx2, val_idx = next(
        skf_val.split(rest_df, rest_df["label"], groups=rest_df["lesion_id"])
    )
    val_df = rest_df.iloc[val_idx].copy()
    train_df = rest_df.iloc[train_idx2].copy()

    # 누수 검증
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    overlaps = (
        (train_lesions & val_lesions)
        | (train_lesions & test_lesions)
        | (val_lesions & test_lesions)
    )
    if overlaps:
        raise RuntimeError(
            f"Lesion ID 누수 발견: {len(overlaps)}건 — split 로직 버그"
        )
    logger.info(
        f"ISIC lesion-level split — train lesions={len(train_lesions)}, "
        f"val={len(val_lesions)}, test={len(test_lesions)} (no overlap)"
    )
    return train_df, val_df, test_df


def build_isic_manifest(isic_dir: str, logger=None):
    if logger is None:
        logger = get_logger("downloader")
    isic_dir = Path(isic_dir)
    gt_csv = isic_dir / "raw" / "ISIC_2019_Training_GroundTruth.csv"
    meta_csv = isic_dir / "raw" / "ISIC_2019_Training_Metadata.csv"
    img_dir = isic_dir / "images"

    if not gt_csv.exists():
        raise FileNotFoundError(gt_csv)

    df = pd.read_csv(gt_csv)
    # 컬럼: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    # UNK=1인 행 제거, SCC는 7-class에서 제외 (필요 시 확장)
    if "UNK" in df.columns:
        df = df[df["UNK"] == 0]
    df = df[(df[ISIC_CLASSES].sum(axis=1) == 1)]  # 단일 라벨만
    df["label"] = df[ISIC_CLASSES].values.argmax(axis=1)
    df["filepath"] = df["image"].apply(lambda n: str(img_dir / f"{n}.jpg"))
    df = df[df["filepath"].apply(lambda p: Path(p).exists())]

    # lesion_id 기반 grouped split 시도. 메타데이터가 없거나 lesion_id가
    # 누락된 경우만 image-level fallback (경고 출력).
    use_lesion_split = False
    if meta_csv.exists():
        meta_df = pd.read_csv(meta_csv)
        if "lesion_id" in meta_df.columns:
            df = df.merge(meta_df[["image", "lesion_id"]], on="image", how="left")
            # lesion_id 결측은 이미지 자체를 그룹으로 취급 (자기 자신만 묶임)
            df["lesion_id"] = df["lesion_id"].fillna(df["image"])
            use_lesion_split = True

    if use_lesion_split:
        train_df, val_df, test_df = _isic_lesion_group_split(df, logger)
    else:
        from sklearn.model_selection import train_test_split
        logger.warning(
            "ISIC_2019_Training_Metadata.csv 미발견 → image-level split fallback. "
            "동일 환자/병변의 이미지가 split 경계를 가로질러 메트릭이 낙관적으로 "
            "편향될 수 있음. 교육용 목적에 한정."
        )
        train_df, temp = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp, test_size=0.5, stratify=temp["label"], random_state=42
        )

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = d[["filepath", "label"]].reset_index(drop=True)
        out.to_csv(isic_dir / f"{name}.csv", index=False)
        logger.info(f"ISIC {name}: {len(out)}")


# ======================================================================
# 진입점
# ======================================================================

def run(task: str, config: dict, agreed: bool = False, subset: float = 1.0, source: str = "huggingface"):
    print_disclaimer()
    logger = get_logger("downloader", config["paths"]["logs_dir"])
    if task == "nih_cxr_binary":
        return download_nih(
            config["paths"]["nih_dir"],
            subset=subset,
            agreed=agreed,
            source=source,
            logger=logger,
        )
    elif task == "isic_multiclass":
        return download_isic(
            config["paths"]["isic_dir"],
            agreed=agreed,
            logger=logger,
        )
    raise ValueError(f"Unknown task: {task}")
