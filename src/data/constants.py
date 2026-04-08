"""Paths and constants for the HydroShare water-use bundle."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents"
SITES_PATH = DATA_ROOT / "1_SecondaryData/sites.csv"
RAW_DIR = DATA_ROOT / "3_QC_Data"
ORIGINAL_EVENTS_DIR = DATA_ROOT / "4_EventFilesOriginal"
PROCESSED_EVENTS_DIR = DATA_ROOT / "5_EnventFiles_Processed"

PROCESSED_WATER_ROOT = PROJECT_ROOT / "processed/water_events"
AUDIT_DIR = PROCESSED_WATER_ROOT / "audit"
SPLITS_DIR = PROCESSED_WATER_ROOT / "splits"
SEQUENCES_DIR = PROCESSED_WATER_ROOT / "sequences"

# QC sampling interval (seconds)
QC_INTERVAL_S = 4

# Fixed model input length
RESAMPLE_LEN = 128

# Gallons per pulse -> liters (NIST)
GAL_TO_L = 3.785411784

# Rows with these labels are dropped when building events_clean (case-insensitive).
DEFAULT_DROP_LABELS = frozenset({"unknown", "unclassified"})

# After the above, benchmark rows exclude irrigation for the primary multiclass task;
# irrigation-only rows are kept in events_irrigation_only.csv.
BENCHMARK_EXCLUDE_LABELS = frozenset({"irrigation"})
