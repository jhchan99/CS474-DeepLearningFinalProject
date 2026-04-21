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
from src.models.multiscale_cnn import build_multiscale_cnn
from src.training.config import TrainConfig


MODELS_SUPPORTING_METADATA = {"cnn", "multiscale_cnn"}


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
        raise ValueError(f"Unknown model: {cfg.model!r}")


def _resolve_device(requested: str | None) -> torch.device:
    """
    Choose a device that is actually usable for inference.

    On shared login nodes ``torch.cuda.is_available()`` may return True while CUDA
    allocations still fail because no GPU is allocated. In that case fall back to CPU.
    """
    if requested is not None:
        return torch.device(requested)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        # Probe that CUDA is really usable, not just visible.
        torch.empty(1, device="cuda")
        return torch.device("cuda")
    except RuntimeError as exc:
        print(f"[evaluate] CUDA visible but unusable ({exc}); falling back to CPU")
        return torch.device("cpu")


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds = [], []
    for batch in loader:
        if len(batch) == 3:
            x, meta, y = batch
            meta = meta.to(device, non_blocking=True)
        else:
            x, y = batch
            meta = None
        x = x.to(device, non_blocking=True)
        logits = model(x, meta) if meta is not None else model(x)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
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

    device = _resolve_device(args.device)

    label_map = load_label_map(Path(cfg.label_map_path))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes = len(label_names)

    use_meta = bool(cfg.use_metadata_head) and cfg.model in MODELS_SUPPORTING_METADATA

    ds = WaterEventDataset(
        args.split,
        Path(cfg.manifest_path),
        Path(cfg.sequences_dir),
        return_metadata=use_meta,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = _build_model(cfg, n_classes, metadata_dim=ds.metadata_dim if use_meta else 0)
    # Always deserialize on CPU first so evaluation works on login nodes too.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    print(f"Evaluating {args.split} split ({len(ds):,} events) on {device}...")
    y_true, y_pred = evaluate_split(model, loader, device)

    out_dir = Path(cfg.log_dir)
    save_eval_results(out_dir, y_true, y_pred, label_names, split=args.split)

    from src.evaluation.metrics import macro_f1
    mf1 = macro_f1(y_true, y_pred)
    print(f"Macro-F1 ({args.split}): {mf1:.4f}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
