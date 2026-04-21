#!/usr/bin/env python3
"""
Average softmax predictions from multiple trained runs on the test split and
write the ensemble's classification report / confusion matrix / summary into a
new ``runs/ensemble_<timestamp>/logs/`` folder — using the same file format as
``src.evaluate`` so ``scripts/plot_results.py`` works on it unchanged.

Usage (from project root, inside the dl-water conda env):

    # Auto-discover every run under runs/ that has both a config+best.pt
    python scripts/ensemble_predict.py

    # Explicit subset
    python scripts/ensemble_predict.py \\
        --runs runs/cnn_balanced runs/multiscale_cnn_minority runs/gru_balanced

    # Custom output name
    python scripts/ensemble_predict.py --name top3_ensemble
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import WaterEventDataset, load_label_map
from src.evaluate import MODELS_SUPPORTING_METADATA, _build_model, _resolve_device
from src.evaluation.metrics import macro_f1, save_eval_results
from src.training.config import TrainConfig


def _find_config_for_run(run_dir: Path, configs_root: Path) -> Path | None:
    """Guess the config path from a run directory name."""
    guess = configs_root / f"train_{run_dir.name}.yaml"
    if guess.exists():
        return guess
    for yaml in configs_root.glob("*.yaml"):
        try:
            cfg = TrainConfig.from_yaml(str(yaml))
        except Exception:
            continue
        if cfg.run_name == run_dir.name:
            return yaml
    return None


@torch.no_grad()
def _predict_probs(
    cfg: TrainConfig,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    project_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, softmax_probs) for the given run on the given split."""
    cfg = cfg.resolved_paths(project_root)
    label_map = load_label_map(Path(cfg.label_map_path))
    n_classes = len(label_map)

    use_meta = bool(cfg.use_metadata_head) and cfg.model in MODELS_SUPPORTING_METADATA
    ds = WaterEventDataset(
        split,
        Path(cfg.manifest_path),
        Path(cfg.sequences_dir),
        return_metadata=use_meta,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = _build_model(cfg, n_classes, metadata_dim=ds.metadata_dim if use_meta else 0)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 3:
            x, meta, y = batch
            meta = meta.to(device, non_blocking=True)
        else:
            x, y = batch
            meta = None
        x = x.to(device, non_blocking=True)
        logits = model(x, meta) if meta is not None else model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble multiple trained runs via averaged softmax.")
    parser.add_argument("--runs", nargs="*", default=None, help="Run directories to ensemble. Defaults to all under runs/ with a best.pt.")
    parser.add_argument("--runs-root", default="runs", help="Root directory of runs (default runs/).")
    parser.add_argument("--configs-root", default="configs", help="Directory holding matching yaml configs.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--name", default=None, help="Optional ensemble name; default is ensemble_<timestamp>.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]
    device = _resolve_device(args.device)
    configs_root = project_root / args.configs_root

    if args.runs:
        run_dirs = [Path(p) if Path(p).is_absolute() else project_root / p for p in args.runs]
    else:
        runs_root = project_root / args.runs_root
        run_dirs = sorted(
            p for p in runs_root.iterdir()
            if p.is_dir() and (p / "checkpoints" / "best.pt").exists()
        )

    if len(run_dirs) < 2:
        raise SystemExit(f"Need at least 2 runs to ensemble; found {len(run_dirs)}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_name = args.name or f"ensemble_{timestamp}"
    out_dir = project_root / args.runs_root / ensemble_name / "logs"

    print(f"Ensembling {len(run_dirs)} run(s) → {out_dir.relative_to(project_root)}")
    print(f"Device: {device}  Split: {args.split}")

    avg_probs: np.ndarray | None = None
    ref_labels: np.ndarray | None = None
    used: list[str] = []

    for run_dir in run_dirs:
        cfg_path = _find_config_for_run(run_dir, configs_root)
        ckpt_path = run_dir / "checkpoints" / "best.pt"
        if cfg_path is None or not ckpt_path.exists():
            print(f"  [skip] {run_dir.name}: no config or no best.pt")
            continue
        print(f"  [run ] {run_dir.name}  ({cfg_path.name})")
        cfg = TrainConfig.from_yaml(str(cfg_path))
        labels, probs = _predict_probs(cfg, ckpt_path, args.split, device, project_root)

        if avg_probs is None:
            avg_probs = probs
            ref_labels = labels
        else:
            if not np.array_equal(labels, ref_labels):
                raise RuntimeError(f"Label order mismatch in run {run_dir.name}; ensure splits/manifest are consistent.")
            avg_probs = avg_probs + probs

        used.append(run_dir.name)

    if not used or avg_probs is None or ref_labels is None:
        raise SystemExit("No valid runs produced predictions.")

    avg_probs = avg_probs / len(used)
    y_pred = avg_probs.argmax(axis=1)

    label_map = load_label_map(Path(project_root / "processed/water_events/label_encoding_benchmark.json"))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    save_eval_results(out_dir, ref_labels, y_pred, label_names, split=args.split)

    mf1 = macro_f1(ref_labels, y_pred)
    members_path = out_dir / "ensemble_members.txt"
    members_path.write_text("\n".join(used) + "\n", encoding="utf-8")

    print("")
    print(f"Ensemble members ({len(used)}): {', '.join(used)}")
    print(f"{args.split} macro-F1: {mf1:.4f}")
    print(f"Results saved to {out_dir.relative_to(project_root)}")


if __name__ == "__main__":
    main()
