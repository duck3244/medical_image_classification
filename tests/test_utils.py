"""utils.py — 시드/경로/태스크 파라미터 단위 테스트."""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from utils import (
    get_task_params,
    load_config,
    save_json,
    set_seed,
)


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(3).tolist())
    set_seed(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_save_and_load_json(tmp_path: Path):
    target = tmp_path / "sub" / "out.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": "한글"}
    save_json(payload, str(target))
    assert target.exists()
    with open(target, encoding="utf-8") as f:
        assert json.load(f) == payload


def test_get_task_params_nih():
    cfg = {
        "task": "nih_cxr_binary",
        "img_size_nih": 256,
        "num_classes_nih": 2,
        "paths": {"nih_dir": "/tmp/nih"},
    }
    tp = get_task_params(cfg)
    assert tp["is_binary"] is True
    assert tp["num_classes"] == 2
    assert tp["img_size"] == 256
    assert tp["class_names"] == ["No_Pneumonia", "Pneumonia"]


def test_get_task_params_isic():
    cfg = {
        "task": "isic_multiclass",
        "img_size_isic": 224,
        "num_classes_isic": 7,
        "isic_classes": ["a", "b", "c", "d", "e", "f", "g"],
        "paths": {"isic_dir": "/tmp/isic"},
    }
    tp = get_task_params(cfg)
    assert tp["is_binary"] is False
    assert tp["num_classes"] == 7
    assert len(tp["class_names"]) == 7


def test_get_task_params_unknown():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task_params({"task": "bogus"})
