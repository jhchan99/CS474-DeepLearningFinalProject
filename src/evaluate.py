#!/usr/bin/env python3
"""
CLI entrypoint for final evaluation of a trained classifier on the test split.

Usage::

    python -m src.evaluate --config configs/train_gru.yaml --checkpoint runs/gru_baseline/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import WaterEventDataset, load_label_map
from src.evaluation.metrics import save_eval_results
from src.models.cnn_bilstm import build_cnn_bilstm
from src.models.cnn_classifier import build_cnn
from src.models.gru_classifier import build_gru
from src.training.config import TrainConfig


def _build_model(cfg: TrainConfig, num_classes: int) -> torch.nn.Module:
    if cfg.model == "gru":
        return build_gru(num_classes, cfg)
    elif cfg.model == "cnn":
        return build_cnn(num_classes, cfg)
    elif cfg.model == "cnn_bilstm":
        return build_cnn_bilstm(num_classes, cfg)
    else:
        raise ValueError(f"Unknown model: {cfg.model!r}")


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on the test split.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the best.pt checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]
    cfg = cfg.resolved_paths(project_root)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_label_map(Path(cfg.label_map_path))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes = len(label_names)

    model = _build_model(cfg, n_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    ds = WaterEventDataset(args.split, Path(cfg.manifest_path), Path(cfg.sequences_dir))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print(f"Evaluating {args.split} split ({len(ds):,} events)…")
    y_true, y_pred = evaluate_split(model, loader, device)

    out_dir = Path(cfg.log_dir)
    save_eval_results(out_dir, y_true, y_pred, label_names, split=args.split)

    from src.evaluation.metrics import macro_f1
    mf1 = macro_f1(y_true, y_pred)
    print(f"Macro-F1 ({args.split}): {mf1:.4f}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
