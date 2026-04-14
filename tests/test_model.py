"""model.py — 백본 빌드/freeze 전이/파라미터 카운트 테스트.

빠른 실행을 위해 MobileNetV2 (가장 가벼움)만 사용한다.
"""
import pytest
import tensorflow as tf

from model import (
    build_model,
    count_params,
    get_preprocess_fn,
    unfreeze_for_finetune,
)


@pytest.fixture(scope="module")
def small_model():
    model, base = build_model(
        backbone="MobileNetV2",
        num_classes=2,
        img_size=96,
        dropout=0.3,
        l2_reg=1e-4,
        weights=None,  # imagenet 다운로드 회피
    )
    return model, base


def test_build_model_binary_output_shape(small_model):
    model, _ = small_model
    out = model(tf.zeros((2, 96, 96, 3)), training=False)
    assert out.shape == (2, 1)
    assert tf.reduce_all(out >= 0.0) and tf.reduce_all(out <= 1.0)


def test_build_model_multiclass_output_shape():
    model, _ = build_model(
        backbone="MobileNetV2",
        num_classes=5,
        img_size=96,
        weights=None,
    )
    out = model(tf.zeros((2, 96, 96, 3)), training=False)
    assert out.shape == (2, 5)
    sums = tf.reduce_sum(out, axis=-1).numpy()
    for s in sums:
        assert abs(s - 1.0) < 1e-4


def test_stage1_base_frozen(small_model):
    _, base = small_model
    assert base.trainable is False


def test_unfreeze_for_finetune_increases_trainable(small_model):
    model, base = small_model
    before = count_params(model)["trainable"]
    info = unfreeze_for_finetune(base, fine_tune_at=50, freeze_bn=True)
    after = count_params(model)["trainable"]
    assert after > before
    assert info["trainable_layers"] > 0
    assert info["fine_tune_at"] == 50


def test_unfreeze_clamps_out_of_range(small_model):
    _, base = small_model
    info = unfreeze_for_finetune(base, fine_tune_at=99999, freeze_bn=True)
    assert info["fine_tune_at"] < info["total_layers"]


def test_get_preprocess_fn_known():
    for bb in ["EfficientNetB0", "MobileNetV2", "ResNet50"]:
        assert callable(get_preprocess_fn(bb))


def test_get_preprocess_fn_unknown():
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_preprocess_fn("FakeNet999")


def test_build_model_unknown_backbone():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model(backbone="FakeNet999", num_classes=2, img_size=96, weights=None)
