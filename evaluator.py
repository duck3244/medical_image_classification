"""
evaluator.py
의료영상 특화 평가 모듈.

- 이진(NIH Pneumonia): AUROC, AUPRC, 민감도/특이도, Youden's J 최적 임계값,
  ROC/PR 커브 PNG, 혼동행렬
- 다중(ISIC 7-class): per-class precision/recall/F1, macro/weighted F1,
  혼동행렬 히트맵, per-class ROC
- 모든 결과 이미지에 Disclaimer 워터마크 삽입
- classification_report.json + metrics_summary.md 저장
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_processor import build_pipeline, load_manifest
from model import build_model
from utils import DISCLAIMER_KO, get_task_params, save_json


# ======================================================================
# 워터마크
# ======================================================================

def _add_disclaimer(fig):
    fig.text(
        0.5, 0.01,
        "교육·학습용 / NOT FOR DIAGNOSIS — " + DISCLAIMER_KO[:60] + "...",
        ha="center", fontsize=7, color="gray",
    )


def _save_fig(fig, path: Path):
    _add_disclaimer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 추론 → y_true, y_prob 수집
# ======================================================================

def _collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset):
    y_true, y_prob = [], []
    for x, y in ds:
        p = model(x, training=False)
        y_true.append(np.asarray(y))
        y_prob.append(p.numpy() if hasattr(p, "numpy") else np.asarray(p))
    return np.concatenate(y_true, axis=0), np.concatenate(y_prob, axis=0)


# ======================================================================
# 이진 평가
# ======================================================================

def _evaluate_binary(y_true, y_prob, out_dir: Path, class_names) -> dict:
    y_true = y_true.reshape(-1).astype(int)
    y_prob = y_prob.reshape(-1).astype(float)

    # AUROC/AUPRC (단일 클래스 방어)
    single_class = len(np.unique(y_true)) < 2
    if single_class:
        auroc = float("nan")
        auprc = float("nan")
        fpr = np.array([0.0, 1.0]); tpr = np.array([0.0, 1.0])
        thr = np.array([0.5])
        prec = np.array([1.0, 1.0]); rec = np.array([0.0, 1.0])
        best_thr = 0.5
        j_idx = 0
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = float(auc(rec, prec))
        j = tpr - fpr
        j_idx = int(np.argmax(j))
        best_thr = float(thr[j_idx])

    y_pred = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # ROC
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUROC={auroc:.3f}" if not np.isnan(auroc) else "AUROC=N/A (single class)")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.scatter([fpr[j_idx]], [tpr[j_idx]], c="red",
               label=f"Best thr={best_thr:.2f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    _save_fig(fig, out_dir / "roc_curve.png")

    # PR
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"AUPRC={auprc:.3f}" if not np.isnan(auprc) else "AUPRC=N/A (single class)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    _save_fig(fig, out_dir / "pr_curve.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (thr={best_thr:.2f})")
    for i in range(2):
        for j_ in range(2):
            ax.text(j_, i, int(cm[i, j_]), ha="center", va="center",
                    color="white" if cm[i, j_] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save_fig(fig, out_dir / "confusion_matrix.png")

    return {
        "task_type": "binary",
        "auroc": auroc,
        "auprc": auprc,
        "best_threshold": best_thr,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


# ======================================================================
# 다중 평가
# ======================================================================

def _evaluate_multiclass(y_true_oh, y_prob, out_dir: Path, class_names) -> dict:
    y_true = y_true_oh.argmax(axis=1)
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        macro_auc = float(roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")

    report = classification_report(
        y_true, y_pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save_fig(fig, out_dir / "confusion_matrix.png")

    # Per-class ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
            cls_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} ({cls_auc:.2f})")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Per-class ROC")
    ax.legend(loc="lower right", fontsize=8)
    _save_fig(fig, out_dir / "roc_per_class.png")

    return {
        "task_type": "multiclass",
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc_ovr": macro_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


# ======================================================================
# 마크다운 요약
# ======================================================================

def _write_markdown_summary(metrics: dict, out_path: Path, task: str, weights: str):
    lines = [
        "# Evaluation Summary",
        "",
        "> **DISCLAIMER**: 본 결과는 교육·학습 목적이며 의료 진단에 사용할 수 없습니다.",
        "",
        f"- Task: `{task}`",
        f"- Weights: `{weights}`",
        f"- Samples: {metrics.get('n_samples')}",
        "",
    ]
    if metrics["task_type"] == "binary":
        lines += [
            "## Binary Metrics",
            f"- AUROC: **{metrics['auroc']:.4f}**",
            f"- AUPRC: **{metrics['auprc']:.4f}**",
            f"- Best threshold (Youden's J): {metrics['best_threshold']:.3f}",
            f"- Sensitivity: {metrics['sensitivity']:.4f}",
            f"- Specificity: {metrics['specificity']:.4f}",
            f"- F1: {metrics['f1']:.4f}",
            f"- Positive samples: {metrics['n_positive']}",
        ]
    else:
        lines += [
            "## Multiclass Metrics",
            f"- Macro F1: **{metrics['macro_f1']:.4f}**",
            f"- Weighted F1: **{metrics['weighted_f1']:.4f}**",
            f"- Macro AUC (OvR): {metrics['macro_auc_ovr']:.4f}",
            "",
            "### Per-class",
        ]
        rep = metrics["classification_report"]
        for k, v in rep.items():
            if isinstance(v, dict) and "f1-score" in v:
                lines.append(
                    f"- `{k}`: P={v['precision']:.3f} R={v['recall']:.3f} "
                    f"F1={v['f1-score']:.3f} (n={int(v.get('support', 0))})"
                )
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# 진입점
# ======================================================================

def evaluate(config: dict, weights_path: str, logger) -> dict:
    """
    test.csv에 대해 평가 수행.
    Returns: 메트릭 dict, 결과는 results/<task>_<timestamp>/ 하위에 저장.
    """
    from datetime import datetime
    tp = get_task_params(config)
    is_binary = tp["is_binary"]

    # 모델 로드 (custom loss 없이 가중치만 사용)
    model, _ = build_model(
        backbone=config["backbone"],
        num_classes=tp["num_classes"],
        img_size=tp["img_size"],
    )
    model.load_weights(weights_path)
    logger.info(f"weights loaded: {weights_path}")

    # 테스트 파이프라인
    test_csv = Path(tp["data_dir"]) / "test.csv"
    df = load_manifest(test_csv)
    test_ds = build_pipeline(
        df,
        task=tp["task"],
        backbone=config["backbone"],
        img_size=tp["img_size"],
        num_classes=tp["num_classes"],
        batch_size=int(config["batch_size"]),
        shuffle=False,
        augment=False,
        use_clahe=bool(config.get("use_clahe", True)),
    )

    y_true, y_prob = _collect_predictions(model, test_ds)
    logger.info(f"predictions collected: y_true={y_true.shape} y_prob={y_prob.shape}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config["paths"]["results_dir"]) / f"{tp['task']}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_binary:
        metrics = _evaluate_binary(y_true, y_prob, out_dir, tp["class_names"])
    else:
        metrics = _evaluate_multiclass(y_true, y_prob, out_dir, tp["class_names"])

    save_json(metrics, str(out_dir / "metrics.json"))
    _write_markdown_summary(metrics, out_dir / "summary.md", tp["task"], weights_path)
    logger.info(f"평가 결과 저장: {out_dir}")
    return {"metrics": metrics, "out_dir": str(out_dir)}
