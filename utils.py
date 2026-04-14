"""
utils.py
공통 유틸리티: Disclaimer, 시드 고정, 로거, GPU/mixed precision 설정, 경로 관리.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ======================================================================
# Disclaimer
# ======================================================================

DISCLAIMER_KO = (
    "본 프로젝트는 개인 학습·연구 목적이며, 실제 의료 진단이나 임상 의사결정에 "
    "사용할 수 없습니다. 의료기기로 승인받지 않았습니다."
)

DISCLAIMER_EN = (
    "This project is for personal educational/research use only. "
    "NOT a medical device. NOT for diagnosis or clinical decisions."
)

DISCLAIMER_BANNER = (
    "=" * 78 + "\n"
    "  [DISCLAIMER / 고지사항]\n"
    "  " + DISCLAIMER_KO + "\n"
    "  " + DISCLAIMER_EN + "\n"
    + "=" * 78
)


def print_disclaimer():
    """모든 CLI 명령 시작 시 호출."""
    print(DISCLAIMER_BANNER, flush=True)


# ======================================================================
# 시드 고정
# ======================================================================

def set_seed(seed: int = 42):
    """재현성을 위한 전역 시드 고정."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


# ======================================================================
# 로거
# ======================================================================

def get_logger(name: str = "medimg", log_dir: str = "logs") -> logging.Logger:
    """파일 + 콘솔 로거."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(Path(log_dir) / f"{name}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ======================================================================
# GPU / Mixed Precision
# ======================================================================

def setup_gpu_memory_growth(logger: logging.Logger = None):
    """
    TF가 전체 VRAM을 선점하지 않도록 memory growth 활성화.
    RTX 4060 8GB 환경에서 필수.
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if logger:
            logger.warning("GPU를 찾지 못했습니다. CPU 모드로 실행됩니다.")
        return False
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if logger:
            logger.info(f"GPU memory growth 활성화: {len(gpus)}개 장치")
        return True
    except RuntimeError as e:
        if logger:
            logger.error(f"Memory growth 설정 실패: {e}")
        return False


def configure_mixed_precision(enable: bool = True, logger: logging.Logger = None):
    """mixed_float16 정책 설정 (VRAM ~40% 절감)."""
    import tensorflow as tf
    if not enable:
        if logger:
            logger.info("Mixed precision 비활성 (fp32)")
        return
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        if logger:
            logger.info("Mixed precision 활성: mixed_float16")
    except Exception as e:
        if logger:
            logger.warning(f"Mixed precision 설정 실패, fp32로 폴백: {e}")


def configure_xla(enable: bool = True, logger: logging.Logger = None):
    """XLA JIT 컴파일 활성화."""
    import tensorflow as tf
    try:
        tf.config.optimizer.set_jit(enable)
        if logger:
            logger.info(f"XLA JIT: {'on' if enable else 'off'}")
    except Exception as e:
        if logger:
            logger.warning(f"XLA 설정 실패: {e}")


def log_vram_usage(logger: logging.Logger, tag: str = ""):
    """현재 VRAM 사용량 로깅 (TF 2.10+)."""
    import tensorflow as tf
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        current_mb = info["current"] / (1024 ** 2)
        peak_mb = info["peak"] / (1024 ** 2)
        logger.info(f"VRAM [{tag}] current={current_mb:.1f}MB peak={peak_mb:.1f}MB")
    except Exception:
        pass


# ======================================================================
# 경로 / 설정
# ======================================================================

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


# ======================================================================
# Config 스키마 검증
# ======================================================================

# (key, expected_type, min_inclusive, max_inclusive). None=무제한.
# bool은 int의 서브타입이므로 _check_types에서 별도 처리.
_CONFIG_SCHEMA = [
    ("task", str, None, None),
    ("backbone", str, None, None),
    ("img_size_nih", int, 32, 1024),
    ("img_size_isic", int, 32, 1024),
    ("num_classes_nih", int, 2, 2),
    ("num_classes_isic", int, 2, 100),
    ("batch_size", int, 1, 4096),
    ("grad_accum_steps", int, 1, 256),
    ("epochs_head", int, 0, 1000),
    ("epochs_finetune", int, 0, 1000),
    ("lr_head", float, 1e-8, 1.0),
    ("lr_finetune", float, 1e-9, 1.0),
    ("fine_tune_at", int, 0, 100000),
    ("dropout", float, 0.0, 0.95),
    ("l2_reg", float, 0.0, 1.0),
    ("loss", str, None, None),
    ("focal_alpha", float, 0.0, 1.0),
    ("focal_gamma", float, 0.0, 10.0),
    ("shuffle_buffer", int, 1, 10**7),
    ("optimizer", str, None, None),
    ("seed", int, 0, 2**31 - 1),
    ("early_stopping_patience", int, 0, 1000),
    ("reduce_lr_patience", int, 0, 1000),
    ("monitor_metric", str, None, None),
]

_CONFIG_BOOL_KEYS = (
    "freeze_bn_on_finetune", "focal_alpha_auto", "class_weight_auto",
    "oversample", "use_clahe", "grayscale_to_rgb", "mixed_precision",
    "xla_jit", "gradient_checkpointing", "memory_growth",
)

_VALID_TASKS = {"nih_cxr_binary", "isic_multiclass"}
_VALID_BACKBONES = {"EfficientNetB0", "MobileNetV2", "ResNet50"}
_VALID_LOSSES = {"focal", "ce"}
_VALID_OPTIMIZERS = {"adamw", "adam", "sgd"}


def validate_config(cfg: dict) -> None:
    """config.json 로드 직후 호출. 잘못된 값이면 ValueError를 raise.

    검증 항목:
    - 필수 키 존재
    - 타입(int/float/str/bool)
    - 수치 범위 (학습률, dropout, batch_size 등)
    - 카테고리 값(task/backbone/loss/optimizer)
    - paths 하위 디렉토리 키
    """
    errors: list = []

    for key, typ, lo, hi in _CONFIG_SCHEMA:
        if key not in cfg:
            errors.append(f"필수 키 누락: '{key}'")
            continue
        v = cfg[key]
        # bool은 int의 서브클래스 → int 키에 True/False가 들어오면 거부
        if typ is int and isinstance(v, bool):
            errors.append(f"'{key}': int 기대 but bool ({v})")
            continue
        if typ is float:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(f"'{key}': number 기대 but {type(v).__name__} ({v!r})")
                continue
            v = float(v)
        elif not isinstance(v, typ):
            errors.append(f"'{key}': {typ.__name__} 기대 but {type(v).__name__} ({v!r})")
            continue
        if lo is not None and v < lo:
            errors.append(f"'{key}'={v} < 최소 {lo}")
        if hi is not None and v > hi:
            errors.append(f"'{key}'={v} > 최대 {hi}")

    for key in _CONFIG_BOOL_KEYS:
        if key in cfg and not isinstance(cfg[key], bool):
            errors.append(f"'{key}': bool 기대 but {type(cfg[key]).__name__}")

    if cfg.get("task") not in _VALID_TASKS:
        errors.append(f"'task'='{cfg.get('task')}' not in {sorted(_VALID_TASKS)}")
    if cfg.get("backbone") not in _VALID_BACKBONES:
        errors.append(f"'backbone'='{cfg.get('backbone')}' not in {sorted(_VALID_BACKBONES)}")
    if cfg.get("loss") not in _VALID_LOSSES:
        errors.append(f"'loss'='{cfg.get('loss')}' not in {sorted(_VALID_LOSSES)}")
    if cfg.get("optimizer", "").lower() not in _VALID_OPTIMIZERS:
        errors.append(f"'optimizer'='{cfg.get('optimizer')}' not in {sorted(_VALID_OPTIMIZERS)}")

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        errors.append("'paths': dict 기대")
    else:
        for k in ("nih_dir", "isic_dir", "models_dir", "logs_dir", "results_dir"):
            if k not in paths:
                errors.append(f"'paths.{k}' 누락")
            elif not isinstance(paths[k], str) or not paths[k]:
                errors.append(f"'paths.{k}': 비어있지 않은 문자열 기대")

    if "isic_classes" in cfg:
        ic = cfg["isic_classes"]
        if not isinstance(ic, list) or not all(isinstance(s, str) for s in ic):
            errors.append("'isic_classes': List[str] 기대")
        elif len(ic) != cfg.get("num_classes_isic", len(ic)):
            errors.append(
                f"'isic_classes' 길이({len(ic)}) ≠ num_classes_isic({cfg.get('num_classes_isic')})"
            )

    if errors:
        raise ValueError(
            "config.json 검증 실패:\n  - " + "\n  - ".join(errors)
        )


def save_json(obj: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_task_params(config: dict) -> dict:
    """현재 task에 맞는 img_size, num_classes 등을 반환."""
    task = config["task"]
    if task == "nih_cxr_binary":
        return {
            "task": task,
            "img_size": config["img_size_nih"],
            "num_classes": config["num_classes_nih"],
            "class_names": ["No_Pneumonia", "Pneumonia"],
            "is_binary": True,
            "data_dir": config["paths"]["nih_dir"],
        }
    elif task == "isic_multiclass":
        return {
            "task": task,
            "img_size": config["img_size_isic"],
            "num_classes": config["num_classes_isic"],
            "class_names": config["isic_classes"],
            "is_binary": False,
            "data_dir": config["paths"]["isic_dir"],
        }
    raise ValueError(f"Unknown task: {task}")


# ======================================================================
# 일괄 초기화
# ======================================================================

def bootstrap(config: dict, logger_name: str = "medimg") -> logging.Logger:
    """
    CLI 진입 직후 한 번 호출. Disclaimer, 시드, GPU, mixed precision, XLA 일괄 설정.
    """
    print_disclaimer()
    ensure_dirs(
        config["paths"]["models_dir"],
        config["paths"]["logs_dir"],
        config["paths"]["results_dir"],
    )
    logger = get_logger(logger_name, config["paths"]["logs_dir"])
    logger.info(f"Task: {config['task']} | Backbone: {config['backbone']}")

    set_seed(config.get("seed", 42))

    if config.get("memory_growth", True):
        setup_gpu_memory_growth(logger)
    configure_mixed_precision(config.get("mixed_precision", True), logger)
    configure_xla(config.get("xla_jit", True), logger)

    return logger
