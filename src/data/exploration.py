from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents"
SITES_PATH = DATA_ROOT / "1_SecondaryData/sites.csv"
LOG_DIR = DATA_ROOT / "2_LogFiles"
RAW_DIR = DATA_ROOT / "3_QC_Data"
ORIGINAL_EVENTS_DIR = DATA_ROOT / "4_EventFilesOriginal"
PROCESSED_EVENTS_DIR = DATA_ROOT / "5_EnventFiles_Processed"
OUTPUT_ROOT = PROJECT_ROOT / "data_exploration"
PLOTS_DIR = OUTPUT_ROOT / "plots"
TABLES_DIR = OUTPUT_ROOT / "tables"
PAPER_SITE_REFERENCE = [
    {"Site": 2, "QCRecordWeeks_Paper": 21.6, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 643.0, "BuildingArea_m2_Paper": 140.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 397.9},
    {"Site": 3, "QCRecordWeeks_Paper": 22.5, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 1408.0, "BuildingArea_m2_Paper": 138.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 234.9},
    {"Site": 4, "QCRecordWeeks_Paper": 9.3, "Occupants_Paper": 4, "IrrigableArea_m2_Paper": 1015.0, "BuildingArea_m2_Paper": 136.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 647.7},
    {"Site": 5, "QCRecordWeeks_Paper": 16.4, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 3118.0, "BuildingArea_m2_Paper": 169.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 1786.0},
    {"Site": 6, "QCRecordWeeks_Paper": 6.2, "Occupants_Paper": 3, "IrrigableArea_m2_Paper": 294.0, "BuildingArea_m2_Paper": 101.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 96.2},
    {"Site": 7, "QCRecordWeeks_Paper": 16.8, "Occupants_Paper": 3, "IrrigableArea_m2_Paper": 241.0, "BuildingArea_m2_Paper": 104.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 55.2},
    {"Site": 8, "QCRecordWeeks_Paper": 6.4, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 1789.0, "BuildingArea_m2_Paper": 160.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 720.6},
    {"Site": 9, "QCRecordWeeks_Paper": 9.3, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 509.0, "BuildingArea_m2_Paper": 173.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 602.4},
    {"Site": 10, "QCRecordWeeks_Paper": 6.2, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 824.0, "BuildingArea_m2_Paper": 102.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 149.0},
    {"Site": 11, "QCRecordWeeks_Paper": 12.4, "Occupants_Paper": 4, "IrrigableArea_m2_Paper": 827.0, "BuildingArea_m2_Paper": 136.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 401.3},
    {"Site": 12, "QCRecordWeeks_Paper": 9.9, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 1744.0, "BuildingArea_m2_Paper": 156.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 181.2},
    {"Site": 13, "QCRecordWeeks_Paper": 7.8, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 742.0, "BuildingArea_m2_Paper": 239.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 1507.5},
    {"Site": 14, "QCRecordWeeks_Paper": 10.4, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 2005.0, "BuildingArea_m2_Paper": 315.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 1099.8},
    {"Site": 15, "QCRecordWeeks_Paper": 9.0, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": 405.0, "BuildingArea_m2_Paper": 171.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 392.2},
    {"Site": 16, "QCRecordWeeks_Paper": 7.4, "Occupants_Paper": 3, "IrrigableArea_m2_Paper": 1162.0, "BuildingArea_m2_Paper": 151.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 247.4},
    {"Site": 17, "QCRecordWeeks_Paper": 8.5, "Occupants_Paper": 3, "IrrigableArea_m2_Paper": 1451.0, "BuildingArea_m2_Paper": 92.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 341.0},
    {"Site": 18, "QCRecordWeeks_Paper": 8.7, "Occupants_Paper": 1, "IrrigableArea_m2_Paper": 410.0, "BuildingArea_m2_Paper": 74.0, "IrrigationMode_Paper": "Hose", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 233.5},
    {"Site": 19, "QCRecordWeeks_Paper": 23.1, "Occupants_Paper": 5, "IrrigableArea_m2_Paper": 982.0, "BuildingArea_m2_Paper": 128.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 854.1},
    {"Site": 20, "QCRecordWeeks_Paper": 5.2, "Occupants_Paper": 4, "IrrigableArea_m2_Paper": 1202.0, "BuildingArea_m2_Paper": 177.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 942.2},
    {"Site": 21, "QCRecordWeeks_Paper": 6.8, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": np.nan, "BuildingArea_m2_Paper": np.nan, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": np.nan},
    {"Site": 22, "QCRecordWeeks_Paper": 8.0, "Occupants_Paper": 7, "IrrigableArea_m2_Paper": np.nan, "BuildingArea_m2_Paper": np.nan, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": np.nan},
    {"Site": 23, "QCRecordWeeks_Paper": 5.7, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": 1108.0, "BuildingArea_m2_Paper": 144.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 809.2},
    {"Site": 24, "QCRecordWeeks_Paper": 8.1, "Occupants_Paper": 8, "IrrigableArea_m2_Paper": 1276.0, "BuildingArea_m2_Paper": 279.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 1308.0},
    {"Site": 25, "QCRecordWeeks_Paper": 7.4, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": 914.0, "BuildingArea_m2_Paper": 282.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 614.4},
    {"Site": 26, "QCRecordWeeks_Paper": 6.7, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": 3592.0, "BuildingArea_m2_Paper": 117.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.0962, "AnnualAverageWaterUse_m3_Paper": 644.0},
    {"Site": 27, "QCRecordWeeks_Paper": 6.7, "Occupants_Paper": 7, "IrrigableArea_m2_Paper": 3842.0, "BuildingArea_m2_Paper": 299.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.0962, "AnnualAverageWaterUse_m3_Paper": 2248.6},
    {"Site": 28, "QCRecordWeeks_Paper": 4.8, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": 1846.0, "BuildingArea_m2_Paper": 337.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 1747.6},
    {"Site": 29, "QCRecordWeeks_Paper": 4.8, "Occupants_Paper": 3, "IrrigableArea_m2_Paper": 700.0, "BuildingArea_m2_Paper": 133.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 573.3},
    {"Site": 30, "QCRecordWeeks_Paper": 5.0, "Occupants_Paper": 6, "IrrigableArea_m2_Paper": 1250.0, "BuildingArea_m2_Paper": 137.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1575, "AnnualAverageWaterUse_m3_Paper": 716.2},
    {"Site": 31, "QCRecordWeeks_Paper": 4.6, "Occupants_Paper": 3, "IrrigableArea_m2_Paper": 827.0, "BuildingArea_m2_Paper": 154.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.1257, "AnnualAverageWaterUse_m3_Paper": 695.7},
    {"Site": 32, "QCRecordWeeks_Paper": 4.6, "Occupants_Paper": 2, "IrrigableArea_m2_Paper": 862.0, "BuildingArea_m2_Paper": 104.0, "IrrigationMode_Paper": "Sprinkler system", "PulseResolution_L_per_pulse_Paper": 0.0329, "AnnualAverageWaterUse_m3_Paper": 730.1},
]


def extract_site_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse site id from {path.name}")
    return int(match.group(1))


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    headers = list(display.columns)

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.2f}".rstrip("0").rstrip(".")
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        return str(value)

    rows = [[format_value(value) for value in row] for row in display.itertuples(index=False, name=None)]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def build_paper_site_reference() -> pd.DataFrame:
    return pd.DataFrame(PAPER_SITE_REFERENCE).sort_values("Site").reset_index(drop=True)


def clean_site_metadata() -> pd.DataFrame:
    sites = pd.read_csv(SITES_PATH)
    sites.columns = (
        sites.columns.str.strip()
        .str.replace("/", "_", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    for column in sites.columns:
        if sites[column].dtype == object:
            cleaned = sites[column].astype(str).str.replace(",", "", regex=False)
            sites[column] = cleaned.where(~sites[column].isna(), np.nan)

    numeric_candidates = [
        "SiteID",
        "N_Residents",
        "N_Residents_0-10",
        "N_Residents_10-25",
        "N_Residents_25-40",
        "N_Residents_40-60",
        "N_Residents_Over60",
        "MeterSize",
        "MeterResolution",
        "N_Bathrooms",
        "Legal_Acreage_SqFt",
        "YearBuilt",
        "Building_Sq_Ft",
        "ZipCode",
        "UserPercentile_City_LastYear",
        "MonthlyAverageWinter",
        "MonthlyAverageSummer",
        "Irr_Area",
    ]

    for column in numeric_candidates:
        if column in sites.columns:
            sites[column] = pd.to_numeric(sites[column], errors="coerce")

    return sites


def load_event_files(event_dir: Path, source_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(event_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if "Site" not in frame.columns:
            frame["Site"] = extract_site_id(csv_path)
        frame["Source"] = source_name
        frame["SourceFile"] = csv_path.name

        for column in ["StartTime", "EndTime"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

        numeric_columns = [
            "Duration",
            "OriginalVolume",
            "OriginalFlowRate",
            "Peak_Value",
            "Mode_Value",
            "Site",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {event_dir}")

    return pd.concat(frames, ignore_index=True)


def load_log_summaries() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for csv_path in sorted(LOG_DIR.glob("*.csv")):
        frame = pd.read_csv(csv_path, parse_dates=["StartDate", "EndDate"])
        numeric_columns = [
            "Meter_WaterUse",
            "CIWS-DL_WaterUse",
            "PercentError_Vol",
            "N_ExpectedValues",
            "N_ActualValues",
            "PercentError_Count",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        site_id = extract_site_id(csv_path)
        records.append(
            {
                "Site": site_id,
                "LogPeriods": len(frame),
                "LogStart": frame["StartDate"].min(),
                "LogEnd": frame["EndDate"].max(),
                "MeanPercentErrorVol": frame["PercentError_Vol"].mean(),
                "MeanPercentErrorCount": frame["PercentError_Count"].mean(),
            }
        )

    return pd.DataFrame(records).sort_values("Site").reset_index(drop=True)


def summarize_raw_file(csv_path: Path, chunksize: int = 250_000) -> dict[str, object]:
    site_id = extract_site_id(csv_path)
    total_rows = 0
    total_pulses = 0.0
    zero_rows = 0
    nonzero_rows = 0
    max_pulses = 0.0
    irregular_intervals = 0
    estimated_missing_steps = 0
    start_time = None
    end_time = None
    previous_time = None

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk["Time"] = pd.to_datetime(chunk["Time"], errors="coerce")
        chunk["Pulses"] = pd.to_numeric(chunk["Pulses"], errors="coerce").fillna(0)
        chunk = chunk.dropna(subset=["Time"])

        if chunk.empty:
            continue

        if start_time is None:
            start_time = chunk["Time"].iloc[0]
        end_time = chunk["Time"].iloc[-1]

        pulses = chunk["Pulses"]
        total_rows += len(chunk)
        total_pulses += float(pulses.sum())
        zero_rows += int((pulses == 0).sum())
        nonzero_rows += int((pulses > 0).sum())
        max_pulses = max(max_pulses, float(pulses.max()))

        diffs = chunk["Time"].diff().dt.total_seconds().iloc[1:]
        irregular_intervals += int((diffs != 4).sum())
        estimated_missing_steps += int(np.maximum((diffs[diffs > 4] / 4) - 1, 0).sum())

        if previous_time is not None:
            chunk_start_gap = (chunk["Time"].iloc[0] - previous_time).total_seconds()
            if chunk_start_gap != 4:
                irregular_intervals += 1
                if chunk_start_gap > 4:
                    estimated_missing_steps += int(max((chunk_start_gap / 4) - 1, 0))

        previous_time = chunk["Time"].iloc[-1]

    days_covered = (
        (end_time - start_time).total_seconds() / 86_400
        if start_time is not None and end_time is not None
        else np.nan
    )

    return {
        "Site": site_id,
        "RawRows": total_rows,
        "RawStart": start_time,
        "RawEnd": end_time,
        "DaysCovered": days_covered,
        "TotalPulses": total_pulses,
        "MeanPulses": (total_pulses / total_rows) if total_rows else np.nan,
        "ZeroFraction": (zero_rows / total_rows) if total_rows else np.nan,
        "NonZeroFraction": (nonzero_rows / total_rows) if total_rows else np.nan,
        "MaxPulses": max_pulses,
        "IrregularIntervals": irregular_intervals,
        "EstimatedMissingSteps": estimated_missing_steps,
    }


def build_raw_site_summary() -> pd.DataFrame:
    records = [summarize_raw_file(csv_path) for csv_path in sorted(RAW_DIR.glob("*.csv"))]
    return pd.DataFrame(records).sort_values("Site").reset_index(drop=True)


def build_event_site_summary(events: pd.DataFrame) -> pd.DataFrame:
    summary = (
        events.groupby(["Source", "Site"], as_index=False)
        .agg(
            EventCount=("Label", "size"),
            LabelCardinality=("Label", pd.Series.nunique),
            MeanDuration=("Duration", "mean"),
            MedianDuration=("Duration", "median"),
            MeanVolume=("OriginalVolume", "mean"),
            TotalVolume=("OriginalVolume", "sum"),
            MeanFlowRate=("OriginalFlowRate", "mean"),
        )
        .sort_values(["Source", "Site"])
        .reset_index(drop=True)
    )
    return summary


def save_inventory_summary(
    sites: pd.DataFrame,
    logs: pd.DataFrame,
    original_events: pd.DataFrame,
    processed_events: pd.DataFrame,
    raw_summary: pd.DataFrame,
) -> pd.DataFrame:
    inventory = pd.DataFrame(
        [
            {"Dataset": "sites_metadata", "Files": 1, "Rows": len(sites), "Sites": sites["SiteID"].nunique()},
            {
                "Dataset": "log_files",
                "Files": len(list(LOG_DIR.glob("*.csv"))),
                "Rows": int(logs["LogPeriods"].sum()),
                "Sites": logs["Site"].nunique(),
            },
            {
                "Dataset": "original_events",
                "Files": len(list(ORIGINAL_EVENTS_DIR.glob("*.csv"))),
                "Rows": len(original_events),
                "Sites": original_events["Site"].nunique(),
            },
            {
                "Dataset": "processed_events",
                "Files": len(list(PROCESSED_EVENTS_DIR.glob("*.csv"))),
                "Rows": len(processed_events),
                "Sites": processed_events["Site"].nunique(),
            },
            {
                "Dataset": "raw_qc_data",
                "Files": len(list(RAW_DIR.glob("*.csv"))),
                "Rows": int(raw_summary["RawRows"].sum()),
                "Sites": raw_summary["Site"].nunique(),
            },
        ]
    )
    inventory.to_csv(TABLES_DIR / "inventory_summary.csv", index=False)
    return inventory


def plot_label_distribution(events: pd.DataFrame) -> None:
    label_counts = (
        events.groupby(["Source", "Label"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values(["Source", "Count"], ascending=[True, False])
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(data=label_counts, x="Label", y="Count", hue="Source")
    plt.xticks(rotation=35, ha="right")
    plt.title("Event Label Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "event_label_distribution.png", dpi=200)
    plt.close()


def plot_duration_and_volume_by_label(processed_events: pd.DataFrame) -> None:
    filtered = processed_events.copy()
    filtered = filtered[
        (filtered["Duration"] > 0)
        & (filtered["OriginalVolume"] > 0)
        & filtered["Label"].notna()
    ]
    filtered["LogDuration"] = np.log10(filtered["Duration"])
    filtered["LogVolume"] = np.log10(filtered["OriginalVolume"])

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=filtered, x="Label", y="LogDuration")
    plt.xticks(rotation=35, ha="right")
    plt.title("Processed Event Duration by Label (log10 minutes)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "processed_duration_by_label.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=filtered, x="Label", y="LogVolume")
    plt.xticks(rotation=35, ha="right")
    plt.title("Processed Event Volume by Label (log10 gallons)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "processed_volume_by_label.png", dpi=200)
    plt.close()


def plot_event_counts_by_site(event_site_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 8))
    sns.barplot(data=event_site_summary, x="Site", y="EventCount", hue="Source")
    plt.xticks(rotation=90)
    plt.title("Event Counts per Site")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "event_counts_by_site.png", dpi=200)
    plt.close()


def plot_raw_site_metrics(raw_summary: pd.DataFrame) -> None:
    ranked = raw_summary.sort_values("ZeroFraction", ascending=False)

    plt.figure(figsize=(14, 8))
    sns.barplot(data=ranked, x="Site", y="ZeroFraction", color="steelblue")
    plt.xticks(rotation=90)
    plt.title("Zero-Pulse Fraction by Site")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "raw_zero_fraction_by_site.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=raw_summary,
        x="DaysCovered",
        y="TotalPulses",
        size="MeanPulses",
        hue="ZeroFraction",
        palette="viridis",
    )
    plt.title("Raw Coverage vs Total Pulses by Site")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "raw_coverage_vs_total_pulses.png", dpi=200)
    plt.close()


def plot_metadata_correlation(merged_site_summary: pd.DataFrame) -> None:
    numeric_columns = [
        "N_Residents",
        "MeterResolution",
        "N_Bathrooms",
        "MonthlyAverageWinter",
        "MonthlyAverageSummer",
        "Irr_Area",
        "EventCount",
        "TotalVolume",
        "DaysCovered",
        "TotalPulses",
        "ZeroFraction",
    ]
    available = [column for column in numeric_columns if column in merged_site_summary.columns]
    corr = merged_site_summary[available].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Metadata, Event, and Raw Summary Correlations")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "site_summary_correlation_heatmap.png", dpi=200)
    plt.close()


def write_report(
    inventory: pd.DataFrame,
    original_events: pd.DataFrame,
    processed_events: pd.DataFrame,
    raw_summary: pd.DataFrame,
    merged_processed_summary: pd.DataFrame,
    paper_site_reference: pd.DataFrame,
) -> None:
    original_label_counts = original_events["Label"].value_counts()
    processed_label_counts = processed_events["Label"].value_counts()
    dominant_processed_label = processed_label_counts.idxmax()
    dominant_processed_share = processed_label_counts.max() / processed_label_counts.sum()

    report_lines = [
        "# Water-Use Dataset Exploration",
        "",
        "## Inventory",
        dataframe_to_markdown(inventory),
        "",
        "## Event Highlights",
        f"- Original event labels: {', '.join(original_label_counts.index.tolist())}",
        f"- Processed event labels: {', '.join(processed_label_counts.index.tolist())}",
        f"- Dominant processed label: `{dominant_processed_label}` ({dominant_processed_share:.1%} of processed events).",
        f"- Median processed duration: {processed_events['Duration'].median():.2f} minutes.",
        f"- Median processed volume: {processed_events['OriginalVolume'].median():.2f} gallons.",
        "",
        "## Raw Sequence Highlights",
        f"- Total raw rows scanned: {int(raw_summary['RawRows'].sum()):,}.",
        f"- Mean site zero-pulse fraction: {raw_summary['ZeroFraction'].mean():.1%}.",
        f"- Raw coverage range: {raw_summary['DaysCovered'].min():.1f} to {raw_summary['DaysCovered'].max():.1f} days.",
        f"- Total estimated missing 4-second steps: {int(raw_summary['EstimatedMissingSteps'].sum()):,}.",
        "",
        "## Metadata Join Highlights",
        (
            "- Highest event-count site: "
            f"{int(merged_processed_summary.sort_values('EventCount', ascending=False).iloc[0]['Site'])} "
            f"with {int(merged_processed_summary['EventCount'].max()):,} processed events."
        ),
        (
            "- Highest total-pulse site: "
            f"{int(merged_processed_summary.sort_values('TotalPulses', ascending=False).iloc[0]['Site'])} "
            f"with {merged_processed_summary['TotalPulses'].max():,.0f} pulses."
        ),
        "",
        "## Paper Table 2 Site Reference",
        "- Added as a supplemental site lookup keyed by `Site` from the paper-provided table.",
        "- Kept separate from `sites.csv` columns because some values differ between the paper table and HydroShare metadata.",
        dataframe_to_markdown(paper_site_reference),
        "",
        "Plots are saved under `data_exploration/plots/` and summary tables under `data_exploration/tables/`.",
    ]

    (OUTPUT_ROOT / "summary_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    print("Loading site metadata...")
    sites = clean_site_metadata()
    paper_site_reference = build_paper_site_reference()

    print("Loading event files...")
    original_events = load_event_files(ORIGINAL_EVENTS_DIR, "original")
    processed_events = load_event_files(PROCESSED_EVENTS_DIR, "processed")
    all_events = pd.concat([original_events, processed_events], ignore_index=True)

    print("Loading log summaries...")
    logs = load_log_summaries()

    print("Summarizing raw QC files...")
    raw_summary = build_raw_site_summary()

    print("Building summary tables...")
    inventory = save_inventory_summary(sites, logs, original_events, processed_events, raw_summary)
    event_site_summary = build_event_site_summary(all_events)
    processed_site_summary = (
        event_site_summary[event_site_summary["Source"] == "processed"]
        .drop(columns=["Source"])
        .reset_index(drop=True)
    )
    merged_processed_summary = (
        sites.merge(processed_site_summary, left_on="SiteID", right_on="Site", how="left")
        .merge(paper_site_reference, on="Site", how="left")
        .merge(raw_summary, on="Site", how="left")
        .merge(logs, on="Site", how="left")
    )

    event_site_summary.to_csv(TABLES_DIR / "event_site_summary.csv", index=False)
    paper_site_reference.to_csv(TABLES_DIR / "paper_site_reference.csv", index=False)
    raw_summary.to_csv(TABLES_DIR / "raw_site_summary.csv", index=False)
    merged_processed_summary.to_csv(TABLES_DIR / "merged_site_summary.csv", index=False)

    print("Saving plots...")
    plot_label_distribution(all_events)
    plot_duration_and_volume_by_label(processed_events)
    plot_event_counts_by_site(event_site_summary)
    plot_raw_site_metrics(raw_summary)
    plot_metadata_correlation(merged_processed_summary)

    print("Writing markdown summary...")
    write_report(
        inventory=inventory,
        original_events=original_events,
        processed_events=processed_events,
        raw_summary=raw_summary,
        merged_processed_summary=merged_processed_summary,
        paper_site_reference=paper_site_reference,
    )

    print(f"Done. Outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()