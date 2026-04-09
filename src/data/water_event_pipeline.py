"""
Build cleaned event tables, site splits, and sequence tensors from original labelled events + QC data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import (
    AUDIT_DIR,
    BENCHMARK_EXCLUDE_LABELS,
    DEFAULT_DROP_LABELS,
    GAL_TO_L,
    ORIGINAL_EVENTS_DIR,
    PROCESSED_WATER_ROOT,
    QC_INTERVAL_S,
    RAW_DIR,
    RESAMPLE_LEN,
    SEQUENCES_DIR,
    SPLITS_DIR,
    SITES_PATH,
)


def ensure_dirs() -> None:
    for d in (PROCESSED_WATER_ROOT, AUDIT_DIR, SPLITS_DIR, SEQUENCES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def extract_site_id_from_path(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse site id from {path.name}")
    return int(match.group(1))


def list_original_event_files() -> list[Path]:
    files = sorted(ORIGINAL_EVENTS_DIR.glob("LabelledEvents_site_*.csv"))
    if not files:
        raise FileNotFoundError(f"No LabelledEvents_site_*.csv under {ORIGINAL_EVENTS_DIR}")
    return files


def qc_path_for_site(site_id: int) -> Path:
    return RAW_DIR / f"site{site_id:03d}qc_data.csv"


def event_time_to_qc_naive(ts: str | pd.Timestamp) -> pd.Timestamp:
    """
    Event files use ISO8601 with trailing Z; QC uses naive timestamps that match the same
    wall-clock values in this bundle. Strip timezone and parse as naive for alignment.
    """
    if isinstance(ts, pd.Timestamp):
        t = ts
    else:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1]
        t = pd.to_datetime(s)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def load_sites_metadata() -> pd.DataFrame:
    df = pd.read_csv(SITES_PATH)
    df["SiteID"] = df["SiteID"].astype(int)
    return df


def load_all_events_concat() -> pd.DataFrame:
    """Load and concatenate all original event CSVs; drop stray index column if present."""
    frames: list[pd.DataFrame] = []
    for path in list_original_event_files():
        site = extract_site_id_from_path(path)
        df = pd.read_csv(path)
        if "X1" in df.columns:
            df = df.drop(columns=["X1"])
        if "Site" not in df.columns:
            df["Site"] = site
        else:
            df["Site"] = df["Site"].fillna(site)
        df["Site"] = df["Site"].astype(int)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["Label"] = out["Label"].astype(str).str.strip()
    return out


def audit_labels(events: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Global label counts and per-site label counts."""
    counts = events["Label"].value_counts()
    per_site = (
        events.groupby(["Site", "Label"], sort=False)
        .size()
        .reset_index(name="count")
    )
    return counts, per_site


def pulses_to_flow_l_per_min(pulses: np.ndarray, meter_resolution_gal_per_pulse: float) -> np.ndarray:
    """Convert per-4s pulse counts to flow rate in L/min."""
    liters_per_bin = pulses.astype(np.float64) * meter_resolution_gal_per_pulse * GAL_TO_L
    return (liters_per_bin * (60.0 / QC_INTERVAL_S)).astype(np.float64)


def resample_linear_1d(y: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate 1D sequence to target_len points."""
    n = len(y)
    if n == 0:
        return np.zeros(target_len, dtype=np.float64)
    if n == 1:
        return np.full(target_len, float(y[0]), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, y.astype(np.float64))


def slice_qc_inclusive_indices(
    time_ns: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[int, int]:
    """Return [i0, i1) slice indices with start <= Time <= end (inclusive on both ends)."""
    st = np.int64(pd.Timestamp(start).value)
    et = np.int64(pd.Timestamp(end).value)
    i0 = int(np.searchsorted(time_ns, st, side="left"))
    i1 = int(np.searchsorted(time_ns, et, side="right"))
    return i0, i1


def build_site_resolution_map(sites: pd.DataFrame) -> dict[int, float]:
    m: dict[int, float] = {}
    for _, row in sites.iterrows():
        m[int(row["SiteID"])] = float(row["MeterResolution"])
    return m


@dataclass
class SplitResult:
    train: list[int]
    val: list[int]
    test: list[int]
    notes: str


def site_split_coverage(
    sites: list[int],
    events: pd.DataFrame,
    val_n: int = 4,
    test_n: int = 4,
    seed: int = 42,
    minority_labels: tuple[str, ...] = ("bathtub", "clotheswasher"),
) -> SplitResult:
    """
    Split sites into train/val/test without mixing sites (~23/4/4 for 31 sites).
    Prefer val and test sets that each contain at least one site with bathtub / clotheswasher events.
    """
    rng = np.random.default_rng(seed)
    sites = sorted(sites)
    assert len(sites) - val_n - test_n > 0

    def sites_with_label(lab: str) -> list[int]:
        s = events.loc[events["Label"].str.lower() == lab.lower(), "Site"].unique()
        return sorted(int(x) for x in s)

    def has_labels(site_set: set[int], labs: tuple[str, ...]) -> dict[str, bool]:
        sub = events[events["Site"].isin(site_set)]
        return {lab: (sub["Label"].str.lower() == lab.lower()).any() for lab in labs}

    assigned_val: set[int] = set()
    assigned_test: set[int] = set()

    for lab in minority_labels:
        cand = [s for s in sites_with_label(lab) if s not in assigned_val and s not in assigned_test]
        rng.shuffle(cand)
        if cand and len(assigned_val) < val_n:
            assigned_val.add(int(cand[0]))

    pool = [s for s in sites if s not in assigned_val and s not in assigned_test]
    rng.shuffle(pool)
    for s in pool:
        if len(assigned_val) >= val_n:
            break
        assigned_val.add(s)

    for lab in minority_labels:
        cand = [
            s
            for s in sites_with_label(lab)
            if s not in assigned_val and s not in assigned_test
        ]
        rng.shuffle(cand)
        if cand and len(assigned_test) < test_n:
            assigned_test.add(int(cand[0]))

    pool = [s for s in sites if s not in assigned_val and s not in assigned_test]
    rng.shuffle(pool)
    for s in pool:
        if len(assigned_test) >= test_n:
            break
        assigned_test.add(s)

    train = [s for s in sites if s not in assigned_val and s not in assigned_test]

    # Size repair (should already be val_n + test_n + train_n)
    while len(assigned_val) < val_n and train:
        assigned_val.add(train.pop())
    while len(assigned_test) < test_n and train:
        assigned_test.add(train.pop())
    while len(assigned_val) > val_n:
        train.append(assigned_val.pop())
    while len(assigned_test) > test_n:
        train.append(assigned_test.pop())

    val = sorted(assigned_val)
    test = sorted(assigned_test)
    train = sorted(train)

    # Refine: try to ensure minority labels appear in val and test (swap with train)
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    for _, split_set in (("val", val_set), ("test", test_set)):
        present = has_labels(split_set, minority_labels)
        for lab in minority_labels:
            if present.get(lab):
                continue
            # site in train that has this label
            donors = [
                s
                for s in train_set
                if s in sites_with_label(lab)
            ]
            rng.shuffle(donors)
            if not donors:
                continue
            d = donors[0]
            # swap donor with a recipient in split that does not have only rare labels - any site
            recipients = list(split_set - {d})
            if not recipients:
                continue
            r = int(rng.choice(recipients))
            split_set.remove(r)
            split_set.add(d)
            train_set.remove(d)
            train_set.add(r)

    train, val, test = sorted(train_set), sorted(val_set), sorted(test_set)

    cov_val = has_labels(val_set, minority_labels)
    cov_test = has_labels(test_set, minority_labels)
    notes_parts = [
        f"seed={seed}, val_n={val_n}, test_n={test_n}",
        f"train_sites={len(train)}, val_sites={len(val)}, test_sites={len(test)}",
        f"minority_in_val={cov_val}",
        f"minority_in_test={cov_test}",
    ]
    return SplitResult(train=train, val=val, test=test, notes="; ".join(str(x) for x in notes_parts))


def verify_split_label_coverage(
    events: pd.DataFrame,
    split_result: SplitResult,
    labels_of_interest: tuple[str, ...],
) -> pd.DataFrame:
    """Rows: label, split, event_count."""
    rows: list[dict[str, Any]] = []
    for split_name, site_list in (
        ("train", split_result.train),
        ("val", split_result.val),
        ("test", split_result.test),
    ):
        sub = events[events["Site"].isin(site_list)]
        for lab in labels_of_interest:
            c = int((sub["Label"].str.lower() == lab.lower()).sum())
            rows.append({"split": split_name, "label": lab, "event_count": c})
    return pd.DataFrame(rows)


def run_alignment_spot_check(site_id: int = 2, n_events: int = 5) -> dict[str, Any]:
    """Load QC + first events; verify naive time alignment and inclusive slice length."""
    path = qc_path_for_site(site_id)
    qc = pd.read_csv(path, parse_dates=["Time"])
    ev_path = ORIGINAL_EVENTS_DIR / f"LabelledEvents_site_{site_id:03d}.csv"
    ev = pd.read_csv(ev_path)
    if "X1" in ev.columns:
        ev = ev.drop(columns=["X1"])
    samples = []
    meter = load_sites_metadata()
    res_map = build_site_resolution_map(meter)
    mr = res_map[site_id]

    for i in range(min(n_events, len(ev))):
        row = ev.iloc[i]
        st = event_time_to_qc_naive(row["StartTime"])
        et = event_time_to_qc_naive(row["EndTime"])
        mask = (qc["Time"] >= st) & (qc["Time"] <= et)
        sub = qc.loc[mask]
        pulses = sub["Pulses"].to_numpy(dtype=np.float64)
        flow = pulses_to_flow_l_per_min(pulses, mr)
        samples.append(
            {
                "event_index": i,
                "start_naive": str(st),
                "end_naive": str(et),
                "n_qc_rows": int(len(sub)),
                "duration_min": float(row["Duration"]),
            }
        )
    return {"site_id": site_id, "samples": samples}


def write_label_audit(events: pd.DataFrame) -> None:
    ensure_dirs()
    counts, per_site = audit_labels(events)
    counts.rename_axis("Label").reset_index(name="count").to_csv(
        AUDIT_DIR / "label_counts_global.csv", index=False
    )
    per_site.to_csv(AUDIT_DIR / "label_counts_per_site.csv", index=False)
    vocab = sorted(events["Label"].unique())
    (AUDIT_DIR / "label_vocabulary.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")


def apply_label_policy(
    events: pd.DataFrame,
    drop_labels: frozenset[str] = DEFAULT_DROP_LABELS,
    benchmark_exclude: frozenset[str] = BENCHMARK_EXCLUDE_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter labels for downstream tables.

    Source rows are original event CSVs under ``4_EventFilesOriginal``.

    - **cleaned**: drops ``drop_labels`` (default: unknown, unclassified), case-insensitive.
    - **benchmark**: cleaned rows excluding ``benchmark_exclude`` (default: irrigation).
      Use for the primary multiclass task; label ids are in ``label_encoding_benchmark.json``.
    - **irrigation_only**: rows labeled irrigation from the cleaned set (auxiliary / separate task).

    Returns:
      - events_all_clean: after dropping unknown/unclassified
      - events_benchmark: also excluding irrigation (primary multiclass task)
      - events_irrigation_only: irrigation rows only (auxiliary)
    """
    lab_lower = events["Label"].str.lower()
    mask_drop = lab_lower.isin({x.lower() for x in drop_labels})
    cleaned = events.loc[~mask_drop].copy()

    bench_mask = ~cleaned["Label"].str.lower().isin({x.lower() for x in benchmark_exclude})
    benchmark = cleaned.loc[bench_mask].copy()
    irrigation_only = cleaned.loc[cleaned["Label"].str.lower() == "irrigation"].copy()

    return cleaned, benchmark, irrigation_only


def encode_labels(labels: pd.Series) -> tuple[dict[str, int], np.ndarray]:
    uniq = sorted(labels.unique())
    str_to_idx = {str(u): i for i, u in enumerate(uniq)}
    idx = labels.map(lambda x: str_to_idx[str(x)]).to_numpy(dtype=np.int32)
    return str_to_idx, idx


def process_all_sequences(
    events_benchmark: pd.DataFrame,
    split_result: SplitResult,
    resolution_map: dict[int, float],
    shard_size: int = 10_000,
    label_to_idx: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """
    For each event in events_benchmark, slice QC, compute flow (L/min), resample to 128 steps.
    Writes NPZ shards (per split) and returns manifest dataframe with shard filenames.
    """
    ensure_dirs()
    total_events = len(events_benchmark)
    site_to_split: dict[int, str] = {}
    for s in split_result.train:
        site_to_split[s] = "train"
    for s in split_result.val:
        site_to_split[s] = "val"
    for s in split_result.test:
        site_to_split[s] = "test"

    str_to_idx = label_to_idx if label_to_idx is not None else encode_labels(events_benchmark["Label"])[0]
    shard_counters = {"train": 0, "val": 0, "test": 0}

    manifest_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush_current() -> None:
        nonlocal current
        if not current or not current["event_id"]:
            current = None
            return
        sp = current["split"]
        idx = shard_counters[sp]
        path = SEQUENCES_DIR / f"{sp}_shard_{idx:04d}.npz"
        np.savez_compressed(
            path,
            event_id=np.array(current["event_id"], dtype=object),
            site=np.array(current["site"], dtype=np.int32),
            label=np.array(current["label"], dtype=np.int32),
            label_str=np.array(current["label_str"], dtype=object),
            flow_128=np.stack(current["flow_128"], axis=0).astype(np.float32),
            orig_len=np.array(current["orig_len"], dtype=np.int32),
            flow_orig=np.array(current["flow_orig"], dtype=object),
        )
        shard_counters[sp] = idx + 1
        print(
            f"[sequences] wrote {path.name} with {len(current['event_id']):,} events",
            flush=True,
        )
        for meta in current["meta"]:
            meta["shard"] = path.name
            manifest_rows.append(meta)
        current = None

    global_eid = 0
    print(f"[sequences] starting sequence generation for {total_events:,} benchmark events", flush=True)
    for site_id in sorted(events_benchmark["Site"].unique()):
        if site_id not in site_to_split:
            continue
        split = site_to_split[site_id]
        sub_ev = events_benchmark[events_benchmark["Site"] == site_id].reset_index(drop=True)
        mr = resolution_map[site_id]
        print(
            f"[sequences] site {site_id} -> split={split}, events={len(sub_ev):,}",
            flush=True,
        )

        qc = (
            pd.read_csv(qc_path_for_site(site_id), parse_dates=["Time"])
            .sort_values("Time")
            .reset_index(drop=True)
        )
        time_ns = qc["Time"].values.astype("datetime64[ns]").astype(np.int64)

        for j in range(len(sub_ev)):
            if current is not None and current["split"] != split:
                flush_current()
            elif current is not None and len(current["event_id"]) >= shard_size:
                flush_current()

            row = sub_ev.iloc[j]
            st = event_time_to_qc_naive(row["StartTime"])
            et = event_time_to_qc_naive(row["EndTime"])
            i0, i1 = slice_qc_inclusive_indices(time_ns, st, et)
            pulses = qc["Pulses"].to_numpy(dtype=np.float64)[i0:i1]
            flow_orig = pulses_to_flow_l_per_min(pulses, mr)
            f128 = resample_linear_1d(flow_orig, RESAMPLE_LEN).astype(np.float32)
            lab_str = str(row["Label"])
            lab_i = str_to_idx[lab_str]
            global_eid += 1
            eid = f"e{global_eid:07d}_s{site_id}"

            meta = {
                "event_id": eid,
                "site": site_id,
                "split": split,
                "label": lab_str,
                "label_idx": lab_i,
                "start_time": str(st),
                "end_time": str(et),
                "orig_len": len(flow_orig),
                "shard": None,
            }

            if current is None:
                current = {
                    "split": split,
                    "event_id": [],
                    "site": [],
                    "label": [],
                    "label_str": [],
                    "flow_128": [],
                    "orig_len": [],
                    "flow_orig": [],
                    "meta": [],
                }
            current["event_id"].append(eid)
            current["site"].append(site_id)
            current["label"].append(lab_i)
            current["label_str"].append(lab_str)
            current["flow_128"].append(f128)
            current["orig_len"].append(len(flow_orig))
            current["flow_orig"].append(flow_orig.astype(np.float32))
            current["meta"].append(meta)

            if global_eid % 10_000 == 0:
                print(
                    f"[sequences] processed {global_eid:,}/{total_events:,} events",
                    flush=True,
                )

            if len(current["event_id"]) >= shard_size:
                flush_current()

    flush_current()
    print(f"[sequences] completed sequence generation: {global_eid:,} events", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    shard_files = sorted(SEQUENCES_DIR.glob("*_shard_*.npz"))
    return manifest, shard_files


def build_full_pipeline(seed: int = 42, max_events: int | None = None) -> None:
    ensure_dirs()
    print(f"[pipeline] starting build_full_pipeline seed={seed} max_events={max_events}", flush=True)

    # 1) Load + audit
    events = load_all_events_concat()
    print(f"[pipeline] loaded {len(events):,} raw labeled events", flush=True)
    write_label_audit(events)

    cleaned, benchmark, irrigation_only = apply_label_policy(events)
    print(
        "[pipeline] label policy results: "
        f"cleaned={len(cleaned):,}, benchmark={len(benchmark):,}, irrigation_only={len(irrigation_only):,}",
        flush=True,
    )
    cleaned.to_csv(PROCESSED_WATER_ROOT / "events_clean.csv", index=False)
    benchmark.to_csv(PROCESSED_WATER_ROOT / "events_benchmark.csv", index=False)
    irrigation_only.to_csv(PROCESSED_WATER_ROOT / "events_irrigation_only.csv", index=False)

    label_map, _ = encode_labels(benchmark["Label"])
    benchmark_run = (
        benchmark.head(max_events).copy()
        if max_events is not None and max_events > 0
        else benchmark
    )
    print(f"[pipeline] benchmark events to process this run: {len(benchmark_run):,}", flush=True)
    (PROCESSED_WATER_ROOT / "label_encoding_benchmark.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    # 2) Alignment spot check (machine-readable; same policy as slicing in process_all_sequences)
    spot = run_alignment_spot_check(site_id=2, n_events=5)
    alignment_artifact = {
        "time_policy": (
            "Parse event StartTime/EndTime as naive by stripping Z (matches QC wall clock in bundle). "
            "QC rows included with start <= Time <= end."
        ),
        "spot_check": spot,
    }
    (AUDIT_DIR / "alignment_spot_check.json").write_text(
        json.dumps(alignment_artifact, indent=2), encoding="utf-8"
    )

    # 3) Site split
    sites_df = load_sites_metadata()
    all_sites = sorted(sites_df["SiteID"].tolist())
    split_result = site_split_coverage(all_sites, cleaned, val_n=4, test_n=4, seed=seed)
    print(
        "[pipeline] site split sizes: "
        f"train={len(split_result.train)}, val={len(split_result.val)}, test={len(split_result.test)}",
        flush=True,
    )
    split_payload = {
        "seed": seed,
        "train": split_result.train,
        "val": split_result.val,
        "test": split_result.test,
        "notes": split_result.notes,
    }
    (SPLITS_DIR / f"site_split_seed{seed}.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )

    cov = verify_split_label_coverage(
        cleaned,
        split_result,
        ("bathtub", "clotheswasher", "faucet", "toilet", "shower"),
    )
    cov.to_csv(AUDIT_DIR / "split_label_coverage.csv", index=False)

    cov_bench = verify_split_label_coverage(
        benchmark,
        split_result,
        ("bathtub", "clotheswasher", "faucet", "toilet", "shower"),
    )
    cov_bench.to_csv(AUDIT_DIR / "split_label_coverage_benchmark.csv", index=False)

    # 4) Sequences (benchmark events only)
    res_map = build_site_resolution_map(sites_df)
    manifest, _ = process_all_sequences(
        benchmark_run, split_result, res_map, label_to_idx=label_map
    )
    manifest.to_csv(PROCESSED_WATER_ROOT / "events_manifest.csv", index=False)
    print(f"[pipeline] wrote manifest with {len(manifest):,} rows", flush=True)
    print("[pipeline] build_full_pipeline completed", flush=True)


if __name__ == "__main__":
    build_full_pipeline()
