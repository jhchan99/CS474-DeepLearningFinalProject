#!/usr/bin/env python3
"""
Validate processed water-event artifacts before training.

Reads benchmark CSV, manifest, split JSON, label encoding, and sequence shards;
writes JSON/CSV summaries under processed/water_events/audit/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.constants import (  # noqa: E402
    AUDIT_DIR,
    PROCESSED_WATER_ROOT,
    SEQUENCES_DIR,
    SPLITS_DIR,
)

REQUIRED_SHARD_KEYS = (
    "event_id",
    "site",
    "label",
    "label_str",
    "flow_128",
    "orig_len",
    "flow_orig",
)
MINORITY_CLASSES = ("bathtub", "clotheswasher")


def load_artifacts(seed: int = 42) -> dict[str, Any]:
    bench_path = PROCESSED_WATER_ROOT / "events_benchmark.csv"
    manifest_path = PROCESSED_WATER_ROOT / "events_manifest.csv"
    split_path = SPLITS_DIR / f"site_split_seed{seed}.json"
    enc_path = PROCESSED_WATER_ROOT / "label_encoding_benchmark.json"

    out: dict[str, Any] = {
        "benchmark": pd.read_csv(bench_path),
        "manifest": pd.read_csv(manifest_path),
        "split": json.loads(split_path.read_text(encoding="utf-8")),
        "label_map": json.loads(enc_path.read_text(encoding="utf-8")),
    }
    return out


def benchmark_label_stats(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["Label"].value_counts()
    total = int(vc.sum())
    rows = []
    for lab, c in vc.items():
        rows.append(
            {
                "label": str(lab),
                "count": int(c),
                "pct": round(100.0 * c / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def validate_splits(manifest: pd.DataFrame, split_payload: dict[str, Any]) -> dict[str, Any]:
    train_sites = set(split_payload["train"])
    val_sites = set(split_payload["val"])
    test_sites = set(split_payload["test"])
    all_three = train_sites | val_sites | test_sites
    disjoint = (
        train_sites.isdisjoint(val_sites)
        and train_sites.isdisjoint(test_sites)
        and val_sites.isdisjoint(test_sites)
    )

    wrong_split: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        site = int(row["site"])
        sp = str(row["split"])
        expected = {"train": train_sites, "val": val_sites, "test": test_sites}.get(sp)
        if expected is None or site not in expected:
            wrong_split.append({"site": site, "split_column": sp, "issue": "site not in split list"})

    minority_presence: dict[str, dict[str, bool]] = {}
    for split_name in ("train", "val", "test"):
        sub = manifest[manifest["split"] == split_name]
        minority_presence[split_name] = {
            lab: bool((sub["label"].str.lower() == lab).any()) for lab in MINORITY_CLASSES
        }

    return {
        "train_n_sites": len(train_sites),
        "val_n_sites": len(val_sites),
        "test_n_sites": len(test_sites),
        "sites_disjoint": disjoint,
        "manifest_rows_wrong_split": len(wrong_split),
        "wrong_split_sample": wrong_split[:20],
        "minority_class_present": minority_presence,
    }


def validate_labels(manifest: pd.DataFrame, label_map: dict[str, int]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        lab = str(row["label"])
        idx = int(row["label_idx"])
        if lab not in label_map:
            mismatches.append({"event_id": row["event_id"], "issue": "label not in encoding"})
            continue
        if label_map[lab] != idx:
            mismatches.append(
                {
                    "event_id": row["event_id"],
                    "issue": f"label_idx {idx} != map {label_map[lab]}",
                }
            )

    manifest_labels = set(manifest["label"].unique())
    encoded_labels = set(label_map.keys())
    extra_in_manifest = manifest_labels - encoded_labels
    missing_from_manifest = encoded_labels - manifest_labels

    return {
        "label_idx_mismatch_count": len(mismatches),
        "label_idx_mismatches_sample": mismatches[:50],
        "labels_in_manifest_not_in_encoding": sorted(extra_in_manifest),
        "encoding_labels_not_in_manifest": sorted(missing_from_manifest),
    }


def manifest_vs_benchmark(manifest: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    n_m = len(manifest)
    n_b = len(benchmark)
    return {
        "benchmark_rows": n_b,
        "manifest_rows": n_m,
        "manifest_to_benchmark_ratio": round(n_m / n_b, 6) if n_b else None,
        "rows_only_in_benchmark": n_b - n_m,
        "note": (
            "If manifest_rows < benchmark_rows, sequences were likely built with "
            "--max-events or an incomplete run; re-run build_water_event_dataset without "
            "--max-events for full coverage."
        ),
    }


def audit_shards(manifest: pd.DataFrame, sample_resolve: int = 200) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return shard_integrity dataframe and summary dict."""
    unique_shards = manifest["shard"].dropna().unique()
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)

    for shard_name in sorted(unique_shards):
        path = SEQUENCES_DIR / shard_name
        rec: dict[str, Any] = {
            "shard": shard_name,
            "exists": path.is_file(),
            "arrays_present": None,
            "n_events": None,
            "flow_128_shape_ok": None,
            "orig_len_positive": None,
            "error": None,
        }
        if not path.is_file():
            rows.append(rec)
            continue
        try:
            data = np.load(path, allow_pickle=True)
            keys = set(data.files)
            missing = [k for k in REQUIRED_SHARD_KEYS if k not in keys]
            rec["arrays_present"] = ",".join(sorted(keys))
            if missing:
                rec["error"] = f"missing keys: {missing}"
                rows.append(rec)
                continue
            n = len(data["event_id"])
            rec["n_events"] = n
            f128 = data["flow_128"]
            rec["flow_128_shape_ok"] = f128.ndim == 2 and f128.shape[1] == 128
            ol = data["orig_len"]
            rec["orig_len_positive"] = bool(np.all(ol >= 1))
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
        rows.append(rec)

    shard_df = pd.DataFrame(rows)

    # Resolve random manifest rows to shard entries
    resolve_ok = 0
    resolve_fail: list[dict[str, Any]] = []
    m_sample = manifest.sample(n=min(sample_resolve, len(manifest)), random_state=rng)
    for _, row in m_sample.iterrows():
        path = SEQUENCES_DIR / row["shard"]
        if not path.is_file():
            resolve_fail.append({"event_id": row["event_id"], "issue": "shard missing"})
            continue
        data = np.load(path, allow_pickle=True)
        ids = data["event_id"]
        mask = ids == row["event_id"]
        if not np.any(mask):
            resolve_fail.append({"event_id": row["event_id"], "issue": "event_id not in shard"})
            continue
        j = int(np.flatnonzero(mask)[0])
        if int(data["label"][j]) != int(row["label_idx"]):
            resolve_fail.append(
                {
                    "event_id": row["event_id"],
                    "issue": f"label idx {data['label'][j]} != manifest {row['label_idx']}",
                }
            )
            continue
        if data["flow_128"][j].shape != (128,):
            resolve_fail.append({"event_id": row["event_id"], "issue": "flow_128 shape"})
            continue
        resolve_ok += 1

    resolve_summary = {
        "sampled": len(m_sample),
        "resolved_ok": resolve_ok,
        "failed": len(resolve_fail),
        "failures_sample": resolve_fail[:30],
    }
    return shard_df, resolve_summary


def orig_len_summaries(manifest: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, g in manifest.groupby("split"):
        stats = g["orig_len"].describe(percentiles=[0.25, 0.5, 0.75])
        parts.append(
            {
                "group": f"split={name}",
                "count": int(g["orig_len"].count()),
                "min": stats.get("min"),
                "p25": stats.get("25%"),
                "median": stats.get("50%"),
                "p75": stats.get("75%"),
                "max": stats.get("max"),
                "zero_or_negative": int((g["orig_len"] <= 0).sum()),
            }
        )
    for lab, g in manifest.groupby("label"):
        stats = g["orig_len"].describe(percentiles=[0.25, 0.5, 0.75])
        parts.append(
            {
                "group": f"label={lab}",
                "count": int(g["orig_len"].count()),
                "min": stats.get("min"),
                "p25": stats.get("25%"),
                "median": stats.get("50%"),
                "max": stats.get("max"),
                "zero_or_negative": int((g["orig_len"] <= 0).sum()),
            }
        )
    return pd.DataFrame(parts)


def split_label_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["split", "label"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    art = load_artifacts()
    benchmark = art["benchmark"]
    manifest = art["manifest"]
    split_payload = art["split"]
    label_map = art["label_map"]

    bench_stats = benchmark_label_stats(benchmark)
    split_info = validate_splits(manifest, split_payload)
    label_info = validate_labels(manifest, label_map)
    mb_info = manifest_vs_benchmark(manifest, benchmark)
    shard_df, resolve_summary = audit_shards(manifest)
    orig_df = orig_len_summaries(manifest)
    slc = split_label_counts(manifest)

    summary = {
        "benchmark_total_rows": len(benchmark),
        "benchmark_unique_labels": int(benchmark["Label"].nunique()),
        "benchmark_label_distribution": bench_stats.to_dict(orient="records"),
        "manifest_total_rows": len(manifest),
        "manifest_vs_benchmark": mb_info,
        "split_validation": split_info,
        "label_encoding_validation": label_info,
        "shard_resolution_sample": resolve_summary,
        "shards_with_errors": int(shard_df["error"].notna().sum() if "error" in shard_df.columns else 0),
        "shards_missing_files": int((~shard_df["exists"]).sum()) if "exists" in shard_df.columns else 0,
    }

    (AUDIT_DIR / "processed_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    bench_stats.to_csv(AUDIT_DIR / "benchmark_label_counts.csv", index=False)
    slc.to_csv(AUDIT_DIR / "split_label_counts.csv", index=False)
    orig_df.to_csv(AUDIT_DIR / "orig_len_summary.csv", index=False)
    shard_df.to_csv(AUDIT_DIR / "shard_integrity.csv", index=False)

    print("Wrote:", AUDIT_DIR / "processed_summary.json")
    print("Wrote:", AUDIT_DIR / "benchmark_label_counts.csv")
    print("Wrote:", AUDIT_DIR / "split_label_counts.csv")
    print("Wrote:", AUDIT_DIR / "orig_len_summary.csv")
    print("Wrote:", AUDIT_DIR / "shard_integrity.csv")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
