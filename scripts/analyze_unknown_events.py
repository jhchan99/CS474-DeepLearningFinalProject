#!/usr/bin/env python3
"""
Exploratory analysis: run a trained model on unknown/unclassified events
that were excluded from the benchmark dataset.

Outputs:
  processed/water_events/audit/unknown_predictions.csv
  processed/water_events/audit/unknown_summary.json

Usage::

    python scripts/analyze_unknown_events.py \
        --config configs/train_gru.yaml \
        --checkpoint runs/gru_baseline/checkpoints/best.pt

    # specify split (default: loads ALL sites' unknown events)
    python scripts/analyze_unknown_events.py \
        --config configs/train_gru.yaml \
        --checkpoint runs/gru_baseline/checkpoints/best.pt \
        --sites 2 5 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.constants import (
    AUDIT_DIR,
    DEFAULT_DROP_LABELS,
    ORIGINAL_EVENTS_DIR,
    PROCESSED_WATER_ROOT,
    QC_INTERVAL_S,
    RAW_DIR,
    RESAMPLE_LEN,
    SEQUENCES_DIR,
    SITES_PATH,
)
from src.data.sequence_dataset import load_label_map
from src.data.water_event_pipeline import (
    event_time_to_qc_naive,
    load_all_events_concat,
    pulses_to_flow_l_per_min,
    resample_linear_1d,
    slice_qc_inclusive_indices,
)
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


def load_unknown_events(sites: list[int] | None = None) -> pd.DataFrame:
    """
    Load events whose label is 'unknown' or 'unclassified' (case-insensitive).
    These are the rows dropped by the benchmark label policy.
    """
    df = load_all_events_concat()
    mask = df["Label"].str.lower().isin(DEFAULT_DROP_LABELS)
    df_unk = df[mask].copy()
    if sites:
        df_unk = df_unk[df_unk["Site"].isin(sites)]
    print(f"Unknown/unclassified events: {len(df_unk):,}")
    print(df_unk["Label"].value_counts().to_string())
    return df_unk.reset_index(drop=True)


def build_flow_tensor(
    events: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """
    For each unknown event, slice QC data and produce a (1, 128) flow tensor.

    Returns
    -------
    flow_tensor : (N, 1, 128) float32 torch tensor
    meta        : DataFrame aligned to flow_tensor with event_id, site, label, start_time, end_time
    """
    meter_res = dict(zip(sites_df["SiteID"], sites_df["MeterResolution"]))

    flows: list[np.ndarray] = []
    meta_rows: list[dict] = []

    site_qc_cache: dict[int, pd.DataFrame] = {}

    for i, row in events.iterrows():
        site = int(row["Site"])
        if site not in site_qc_cache:
            # Try standard QC filename patterns
            candidates = list(RAW_DIR.glob(f"site{site:03d}qc_data.csv")) + \
                         list(RAW_DIR.glob(f"site{site}qc_data.csv"))
            if not candidates:
                continue
            qc = pd.read_csv(candidates[0], parse_dates=["Time"])
            site_qc_cache[site] = qc

        qc = site_qc_cache[site]
        t_start = event_time_to_qc_naive(str(row["StartTime"]))
        t_end = event_time_to_qc_naive(str(row["EndTime"]))

        lo, hi = slice_qc_inclusive_indices(qc["Time"].values, t_start, t_end)
        pulses = qc.iloc[lo : hi + 1]["Pulses"].to_numpy(dtype=np.float32)
        if len(pulses) == 0:
            continue

        gal_per_pulse = meter_res.get(site, 0.1)
        flow = pulses_to_flow_l_per_min(pulses, gal_per_pulse, QC_INTERVAL_S)
        flow_128 = resample_linear_1d(flow, RESAMPLE_LEN)

        flows.append(flow_128)
        meta_rows.append({
            "site": site,
            "label": row["Label"],
            "start_time": str(row["StartTime"]),
            "end_time": str(row["EndTime"]),
            "orig_len": len(pulses),
        })

    if not flows:
        raise RuntimeError("No processable unknown events found. Check site QC files are present.")

    tensor = torch.from_numpy(np.stack(flows)).float().unsqueeze(1)  # (N, 1, 128)
    meta = pd.DataFrame(meta_rows)
    return tensor, meta


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    flow_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    model.to(device)
    all_preds: list[np.ndarray] = []
    for i in range(0, len(flow_tensor), batch_size):
        batch = flow_tensor[i : i + batch_size].to(device)
        logits = model(batch)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained model on unknown/unclassified events for exploratory analysis."
    )
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (best.pt).")
    parser.add_argument("--sites", nargs="*", type=int, default=None,
                        help="Limit to specific site IDs. Default: all sites.")
    parser.add_argument("--device", default=None, help="torch device string.")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]
    cfg = TrainConfig.from_yaml(args.config)
    cfg = cfg.resolved_paths(project_root)

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    label_map = load_label_map(Path(cfg.label_map_path))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes = len(label_names)

    # Load model
    model = _build_model(cfg, n_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)

    # Load sites metadata
    sites_df = pd.read_csv(SITES_PATH)

    # Load unknown events
    print("Loading unknown/unclassified events…")
    events = load_unknown_events(args.sites)
    if len(events) == 0:
        print("No unknown events found. Exiting.")
        return

    print("Building flow tensors…")
    flow_tensor, meta = build_flow_tensor(events, sites_df)
    print(f"Processable unknown events: {len(meta):,}")

    print("Running inference…")
    pred_idxs = run_inference(model, flow_tensor, device)
    meta["predicted_label"] = [label_names[i] for i in pred_idxs]
    meta["predicted_idx"] = pred_idxs

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = AUDIT_DIR / "unknown_predictions.csv"
    meta.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")

    # Summary
    pred_counts = meta["predicted_label"].value_counts().to_dict()
    summary = {
        "model": cfg.model,
        "checkpoint": args.checkpoint,
        "total_unknown_events": len(events),
        "processable_events": len(meta),
        "predicted_label_distribution": pred_counts,
    }
    out_json = AUDIT_DIR / "unknown_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {out_json}")
    print(json.dumps(pred_counts, indent=2))


if __name__ == "__main__":
    main()
