#!/usr/bin/env python3
"""
CLI entrypoint for training a water end-use classifier.

Usage::

    python -m src.train --config configs/train_gru.yaml
    python -m src.train --config configs/train_cnn.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data.sequence_dataset import WaterEventDataset, load_label_map
from src.models.cnn_bilstm import build_cnn_bilstm
from src.models.cnn_classifier import build_cnn
from src.models.gru_classifier import build_gru
from src.models.multiscale_cnn import build_multiscale_cnn
from src.training.config import TrainConfig
from src.training.train_loop import train


MODELS_SUPPORTING_METADATA = {"cnn", "multiscale_cnn"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(cfg: TrainConfig, num_classes: int, metadata_dim: int = 0) -> torch.nn.Module:
    if cfg.model == "gru":
        return build_gru(num_classes, cfg)
    elif cfg.model == "cnn":
        return build_cnn(num_classes, cfg, metadata_dim=metadata_dim)
    elif cfg.model == "cnn_bilstm":
        return build_cnn_bilstm(num_classes, cfg)
    elif cfg.model == "multiscale_cnn":
        return build_multiscale_cnn(num_classes, cfg, metadata_dim=metadata_dim)
    else:
        raise ValueError(
            f"Unknown model: {cfg.model!r}. "
            "Choose from: gru, cnn, cnn_bilstm, multiscale_cnn"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a water end-use deep classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--device", default=None, help="torch device string (e.g. cpu, cuda, cuda:1).")
    parser.add_argument("--project-root", default=None, help="Override project root for path resolution.")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]
    cfg = cfg.resolved_paths(project_root)

    _set_seed(cfg.seed)

    device = torch.device(args.device) if args.device else None

    label_map = load_label_map(Path(cfg.label_map_path))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes = len(label_names)

    print(f"Run: {cfg.run_name}  Model: {cfg.model}  Classes: {n_classes}  Device: {device or 'auto'}")

    use_meta = bool(cfg.use_metadata_head) and cfg.model in MODELS_SUPPORTING_METADATA
    if cfg.use_metadata_head and cfg.model not in MODELS_SUPPORTING_METADATA:
        print(
            f"[train] use_metadata_head=True but model {cfg.model!r} does not support a "
            "metadata head; ignoring."
        )

    train_ds = WaterEventDataset(
        "train",
        Path(cfg.manifest_path),
        Path(cfg.sequences_dir),
        augment=cfg.augment,
        aug_noise_std=cfg.aug_noise_std,
        aug_amp_min=cfg.aug_amp_min,
        aug_amp_max=cfg.aug_amp_max,
        aug_time_warp=cfg.aug_time_warp,
        return_metadata=use_meta,
    )
    val_ds = WaterEventDataset(
        "val",
        Path(cfg.manifest_path),
        Path(cfg.sequences_dir),
        return_metadata=use_meta,
    )

    print(f"Train size: {len(train_ds):,}  Val size: {len(val_ds):,}")
    if cfg.augment:
        print(
            f"[train] augment=True (noise_std={cfg.aug_noise_std}, "
            f"amp=[{cfg.aug_amp_min},{cfg.aug_amp_max}], time_warp={cfg.aug_time_warp})"
        )
    if use_meta:
        print(f"[train] metadata head enabled (metadata_dim={train_ds.metadata_dim})")

    model = _build_model(cfg, n_classes, metadata_dim=train_ds.metadata_dim if use_meta else 0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    result = train(model, cfg, train_ds, val_ds, label_names, device=device)

    summary_path = Path(cfg.log_dir) / "train_summary.json"
    summary_path.write_text(json.dumps({
        "run_name": cfg.run_name,
        "model": cfg.model,
        "best_val_f1": result["best_val_f1"],
        "best_epoch": result["best_epoch"],
        "n_params": n_params,
    }, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
