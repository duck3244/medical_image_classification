"""data_processor.py — manifest 로드/class weight/파이프라인 형상 테스트."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from data_processor import (
    _apply_clahe_np,
    build_pipeline,
    compute_class_weights,
    load_manifest,
)


@pytest.fixture
def dummy_manifest(tmp_path: Path):
    rows = []
    for i in range(8):
        p = tmp_path / f"img_{i}.png"
        arr = (np.random.RandomState(i).rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        rows.append({"filepath": str(p), "label": i % 2})
    df = pd.DataFrame(rows)
    csv = tmp_path / "manifest.csv"
    df.to_csv(csv, index=False)
    return csv, df


def test_load_manifest_required_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(str(bad))


def test_load_manifest_label_dtype(dummy_manifest):
    csv, _ = dummy_manifest
    df = load_manifest(str(csv))
    assert df["label"].dtype == np.int32
    assert {"filepath", "label"}.issubset(df.columns)


def test_compute_class_weights_balanced():
    labels = np.array([0, 0, 0, 0, 1, 1])
    cw = compute_class_weights(labels)
    assert set(cw.keys()) == {0, 1}
    assert cw[1] > cw[0]  # 소수 클래스에 더 큰 가중치


def test_apply_clahe_rgb_input():
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = _apply_clahe_np(rgb)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 0], out[..., 1])  # 3채널 복제


def test_build_pipeline_xray_shapes(dummy_manifest):
    _, df = dummy_manifest
    ds = build_pipeline(
        df,
        task="nih_cxr_binary",
        backbone="MobileNetV2",
        img_size=64,
        num_classes=2,
        batch_size=4,
        shuffle=True,
        augment=True,
        use_clahe=True,
    )
    x, y = next(iter(ds))
    assert x.shape == (4, 64, 64, 3)
    assert y.shape == (4, 1)
    assert x.dtype == tf.float32


def test_build_pipeline_skin_shapes(dummy_manifest):
    _, df = dummy_manifest
    df_multi = df.copy()
    df_multi["label"] = [i % 3 for i in range(len(df_multi))]
    ds = build_pipeline(
        df_multi,
        task="isic_multiclass",
        backbone="MobileNetV2",
        img_size=64,
        num_classes=3,
        batch_size=4,
        shuffle=False,
        augment=False,
        use_clahe=False,
    )
    x, y = next(iter(ds))
    assert x.shape[1:] == (64, 64, 3)
    assert y.shape[-1] == 3
    assert tf.reduce_all(tf.reduce_sum(y, axis=-1) == 1.0)
