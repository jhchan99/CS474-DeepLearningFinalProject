#!/usr/bin/env python3
"""
Generate training charts and test-evaluation charts for every model run.

For each run found under ``runs/`` a timestamped sub-folder is created:
    runs/<run_name>/charts/<YYYYMMDD_HHMMSS>/

Charts produced (where data exists):
  loss_curve.png          – train vs val loss per epoch
  f1_curve.png            – train vs val macro-F1 per epoch
  confusion_matrix.png    – normalised confusion matrix  (needs test eval)
  per_class_metrics.png   – per-class precision / recall / F1 bar chart (needs test eval)

If multiple runs have history data a comparison chart is also saved to:
    runs/comparison_<YYYYMMDD_HHMMSS>.png

Usage (from project root, inside the dl-water conda env):
    python scripts/plot_results.py
    python scripts/plot_results.py --runs runs/gru_baseline runs/cnn_baseline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10")
TRAIN_COLOR = PALETTE[0]
VAL_COLOR = PALETTE[1]
FIG_DPI = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(Path.cwd())}")


# ── Per-run charts ─────────────────────────────────────────────────────────────

def plot_loss_curve(history: dict, summary: dict, out_dir: Path) -> None:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = list(range(1, len(train_loss) + 1))
    best_ep = summary.get("best_epoch")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, color=TRAIN_COLOR, linewidth=1.8, label="Train loss")
    ax.plot(epochs, val_loss, color=VAL_COLOR, linewidth=1.8, label="Val loss")

    if best_ep and 1 <= best_ep <= len(epochs):
        ax.axvline(best_ep, color="grey", linestyle="--", linewidth=1.2, label=f"Best epoch ({best_ep})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    run_name = summary.get("run_name", out_dir.parent.parent.name)
    ax.set_title(f"{run_name} — Loss curve")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, out_dir / "loss_curve.png")


def plot_f1_curve(history: dict, summary: dict, out_dir: Path) -> None:
    train_f1 = history["train_f1"]
    val_f1 = history["val_f1"]
    epochs = list(range(1, len(train_f1) + 1))
    best_ep = summary.get("best_epoch")
    best_f1 = summary.get("best_val_f1")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_f1, color=TRAIN_COLOR, linewidth=1.8, label="Train macro-F1")
    ax.plot(epochs, val_f1, color=VAL_COLOR, linewidth=1.8, label="Val macro-F1")

    if best_ep and 1 <= best_ep <= len(epochs):
        ax.axvline(best_ep, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Best epoch {best_ep}  (F1={best_f1:.4f})")
        ax.plot(best_ep, best_f1, "o", color=VAL_COLOR, markersize=8, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(bottom=0)
    run_name = summary.get("run_name", out_dir.parent.parent.name)
    ax.set_title(f"{run_name} — Macro-F1 curve")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, out_dir / "f1_curve.png")


def plot_confusion_matrix(cm_df: pd.DataFrame, run_name: str, out_dir: Path) -> None:
    labels = cm_df.index.tolist()
    cm = cm_df.values.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / np.where(row_sums > 0, row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Recall (row-normalised)"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_name} — Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    _save(fig, out_dir / "confusion_matrix.png")


def plot_per_class_metrics(report: dict, run_name: str, out_dir: Path) -> None:
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [k for k in report if k not in skip]
    if not classes:
        return

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25
    colors = [PALETTE[2], PALETTE[3], PALETTE[4]]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.4), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report[c][metric] for c in classes]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{run_name} — Per-class metrics (test set)")
    ax.legend()

    macro = report.get("macro avg", {})
    if macro:
        ax.axhline(macro.get("f1-score", 0), color="grey", linestyle=":", linewidth=1.2,
                   label=f"Macro-F1 = {macro['f1-score']:.3f}")
        ax.legend()

    _save(fig, out_dir / "per_class_metrics.png")


# ── Comparison chart ───────────────────────────────────────────────────────────

def plot_comparison(run_infos: list[dict], out_path: Path) -> None:
    """Overlay val macro-F1 curves for all runs on a single axes."""
    if len(run_infos) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, info in enumerate(run_infos):
        color = PALETTE[i % len(PALETTE)]
        label = info["run_name"]
        epochs = list(range(1, len(info["val_f1"]) + 1))
        best_ep = info.get("best_epoch")
        best_f1 = info.get("best_val_f1")

        axes[0].plot(epochs, info["val_f1"], color=color, linewidth=1.8, label=label)
        if best_ep:
            axes[0].plot(best_ep, best_f1, "o", color=color, markersize=7, zorder=5)

        axes[1].plot(epochs, info["val_loss"], color=color, linewidth=1.8, label=label)

    for ax, ylabel, title in zip(
        axes,
        ["Val macro-F1", "Val loss"],
        ["Validation macro-F1 — all models", "Validation loss — all models"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Model comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def process_run(run_dir: Path, timestamp: str) -> dict | None:
    logs_dir = run_dir / "logs"
    history = _load_json(logs_dir / "history.json")
    summary = _load_json(logs_dir / "train_summary.json")

    if not history or not summary:
        print(f"  [skip] {run_dir.name}: missing history.json or train_summary.json")
        return None

    out_dir = run_dir / "charts" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = summary.get("run_name", run_dir.name)
    print(f"\n[{run_name}]  →  {out_dir.relative_to(Path.cwd())}/")

    plot_loss_curve(history, summary, out_dir)
    plot_f1_curve(history, summary, out_dir)

    cm_df = _load_csv(logs_dir / "test_confusion_matrix.csv")
    if cm_df is not None:
        plot_confusion_matrix(cm_df, run_name, out_dir)

    report = _load_json(logs_dir / "test_classification_report.json")
    if report:
        plot_per_class_metrics(report, run_name, out_dir)

    return {
        "run_name": run_name,
        "val_f1": history["val_f1"],
        "val_loss": history["val_loss"],
        "best_epoch": summary.get("best_epoch"),
        "best_val_f1": summary.get("best_val_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training results for one or more runs.")
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Paths to run directories. Defaults to all sub-dirs of runs/.",
    )
    parser.add_argument(
        "--runs-root", default="runs",
        help="Root directory to auto-discover runs from (default: runs/).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.runs:
        run_dirs = [Path(p) for p in args.runs]
    else:
        runs_root = project_root / args.runs_root
        run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "logs").exists())

    if not run_dirs:
        print("No run directories found. Did you train a model first?")
        return

    print(f"Found {len(run_dirs)} run(s). Timestamp: {timestamp}")

    all_infos: list[dict] = []
    for run_dir in run_dirs:
        info = process_run(run_dir, timestamp)
        if info:
            all_infos.append(info)

    if len(all_infos) >= 2:
        comp_path = project_root / "runs" / f"comparison_{timestamp}.png"
        print(f"\n[comparison]  →  {comp_path.relative_to(project_root)}")
        plot_comparison(all_infos, comp_path)

    print(f"\nDone. {len(all_infos)} run(s) plotted.")


if __name__ == "__main__":
    main()
