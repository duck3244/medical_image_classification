"""trainer.py — focal loss / build_loss 수치 정확성 테스트."""
import numpy as np
import pytest
import tensorflow as tf

from trainer import (
    binary_focal_loss,
    categorical_focal_loss,
    build_loss,
    build_metrics,
)


def test_binary_focal_loss_perfect_prediction_near_zero():
    loss_fn = binary_focal_loss(alpha=0.25, gamma=2.0)
    y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
    y_pred = tf.constant([[0.999], [0.001], [0.999], [0.001]])
    loss = loss_fn(y_true, y_pred).numpy()
    assert loss < 1e-4


def test_binary_focal_loss_wrong_prediction_large():
    loss_fn = binary_focal_loss(alpha=0.25, gamma=2.0)
    y_true = tf.constant([[1.0], [1.0]])
    y_pred = tf.constant([[0.01], [0.01]])
    loss = loss_fn(y_true, y_pred).numpy()
    assert loss > 0.5


def test_categorical_focal_loss_perfect_prediction_near_zero():
    loss_fn = categorical_focal_loss(alpha=0.25, gamma=2.0)
    y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = tf.constant([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005]])
    loss = loss_fn(y_true, y_pred).numpy()
    assert loss < 1e-4


def test_focal_loss_no_nan_at_extremes():
    """clip_by_value 동작 확인: log(0) 방지."""
    loss_fn = binary_focal_loss()
    y_true = tf.constant([[1.0], [0.0]])
    y_pred = tf.constant([[0.0], [1.0]])
    loss = loss_fn(y_true, y_pred).numpy()
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_build_loss_dispatch():
    bin_focal = build_loss(is_binary=True, kind="focal", alpha=0.25, gamma=2.0)
    cat_focal = build_loss(is_binary=False, kind="focal", alpha=0.25, gamma=2.0)
    bin_ce = build_loss(is_binary=True, kind="ce", alpha=0, gamma=0)
    cat_ce = build_loss(is_binary=False, kind="ce", alpha=0, gamma=0)
    assert callable(bin_focal)
    assert callable(cat_focal)
    assert isinstance(bin_ce, tf.keras.losses.BinaryCrossentropy)
    assert isinstance(cat_ce, tf.keras.losses.CategoricalCrossentropy)


def test_build_metrics_binary():
    metrics = build_metrics(is_binary=True)
    names = [m.name for m in metrics]
    assert "auc" in names
    assert "precision" in names
    assert "recall" in names


def test_build_metrics_multiclass():
    metrics = build_metrics(is_binary=False)
    names = [m.name for m in metrics]
    assert "auc" in names
    assert "acc" in names
