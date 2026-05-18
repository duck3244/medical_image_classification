"""utils.validate_config — config 스키마 검증 단위 테스트."""
import copy
import json
from pathlib import Path

import pytest

from utils import load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def base_config():
    return load_config(str(PROJECT_ROOT / "config.json"))


def test_real_config_json_passes(base_config):
    """저장소에 커밋된 config.json 은 항상 검증을 통과해야 한다."""
    validate_config(base_config)


def test_missing_required_key(base_config):
    cfg = copy.deepcopy(base_config)
    del cfg["batch_size"]
    with pytest.raises(ValueError, match="batch_size"):
        validate_config(cfg)


def test_negative_learning_rate(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["lr_head"] = -1e-4
    with pytest.raises(ValueError, match="lr_head"):
        validate_config(cfg)


def test_dropout_out_of_range(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["dropout"] = 1.5
    with pytest.raises(ValueError, match="dropout"):
        validate_config(cfg)


def test_invalid_backbone(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["backbone"] = "FakeNet999"
    with pytest.raises(ValueError, match="backbone"):
        validate_config(cfg)


def test_invalid_task(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["task"] = "bogus_task"
    with pytest.raises(ValueError, match="task"):
        validate_config(cfg)


def test_invalid_loss(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["loss"] = "hinge"
    with pytest.raises(ValueError, match="loss"):
        validate_config(cfg)


def test_bool_in_int_field_rejected(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["batch_size"] = True
    with pytest.raises(ValueError, match="batch_size"):
        validate_config(cfg)


def test_non_bool_in_bool_field(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["mixed_precision"] = "yes"
    with pytest.raises(ValueError, match="mixed_precision"):
        validate_config(cfg)


def test_isic_classes_length_mismatch(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["isic_classes"] = ["MEL", "NV"]  # but num_classes_isic = 7
    with pytest.raises(ValueError, match="isic_classes"):
        validate_config(cfg)


def test_paths_missing_subkey(base_config):
    cfg = copy.deepcopy(base_config)
    del cfg["paths"]["models_dir"]
    with pytest.raises(ValueError, match="models_dir"):
        validate_config(cfg)


def test_grad_accum_steps_zero(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["grad_accum_steps"] = 0
    with pytest.raises(ValueError, match="grad_accum_steps"):
        validate_config(cfg)


def test_load_config_validates(tmp_path: Path, base_config):
    bad = copy.deepcopy(base_config)
    bad["lr_head"] = -1.0
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_config(str(p))
