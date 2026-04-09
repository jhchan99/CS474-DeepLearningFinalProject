"""
Evaluation helpers: macro-F1, per-class metrics, confusion matrix.

All functions accept plain numpy arrays or python lists of ints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Return a dict mirroring sklearn's classification_report, keyed by class name.

    Keys per class: precision, recall, f1-score, support.
    Top-level keys: macro avg, weighted avg, accuracy.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    return report


def confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return (num_classes, num_classes) confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def save_eval_results(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    split: str = "test",
) -> None:
    """Save per-class metrics JSON and confusion matrix CSV to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    report = per_class_report(y_true, y_pred, label_names)
    (out_dir / f"{split}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    cm = confusion(y_true, y_pred, len(label_names))
    header = ",".join(label_names)
    rows = [header]
    for i, row in enumerate(cm):
        rows.append(label_names[i] + "," + ",".join(map(str, row)))
    (out_dir / f"{split}_confusion_matrix.csv").write_text(
        "\n".join(rows), encoding="utf-8"
    )

    summary = {
        "split": split,
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(report["accuracy"]),
    }
    (out_dir / f"{split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
