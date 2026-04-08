---
name: Water Event Preprocessing
overview: Refine the proposed data-prep workflow for sequence modeling on the HydroShare water-use dataset, with special attention to timestamp alignment, site-level splitting, and label policy.
todos:
  - id: audit-label-source
    content: Confirm actual label vocabulary and counts in the chosen event files before setting exclusion rules.
    status: completed
  - id: validate-time-alignment
    content: Define and verify timezone handling and inclusive slice boundaries between event timestamps and QC traces.
    status: in_progress
  - id: design-site-split
    content: Create a reproducible site-level split and verify minority-class presence in validation and test sets.
    status: pending
  - id: define-sequence-outputs
    content: Produce both original-length aligned sequences and 128-step interpolated sequences for downstream modeling.
    status: pending
isProject: false
---

# Refine Water Event Dataset Plan

Your plan is strong, and I would keep the core shape. I would make four important adjustments before implementation.

## What To Keep

- Start from the original event files in `[/home/jamcha99/DeepLearningFinalProject/src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents/4_EventFilesOriginal](file:///home/jamcha99/DeepLearningFinalProject/src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents/4_EventFilesOriginal)`.
- Join site metadata from `[/home/jamcha99/DeepLearningFinalProject/src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents/1_SecondaryData/sites.csv](file:///home/jamcha99/DeepLearningFinalProject/src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents/1_SecondaryData/sites.csv)`.
- Build sequences from raw QC traces in `[/home/jamcha99/DeepLearningFinalProject/src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents/3_QC_Data](file:///home/jamcha99/DeepLearningFinalProject/src/data/0b72cddfc51c45b188e0e6cd8927227e/data/contents/3_QC_Data)`.
- Split strictly by site.
- Check post-split label coverage before training.

## What I Would Add

- Add an explicit timestamp-normalization step before slicing. The original events use UTC-style timestamps like `2019-04-01T14:44:15Z`, while QC traces appear as naive timestamps like `2019-04-01 14:43:15`. This is the biggest risk for silent misalignment.
- Add a slice-boundary rule. The event files appear to encode `StartTime` and `EndTime` as the first and last occupied 4-second bins, so event extraction should likely be inclusive on both ends.
- Add a split-manifest artifact. Save the exact train/val/test site IDs so every experiment uses the same leakage-safe partition.
- Add a trace-quality filter/report. Count events whose raw slice is missing timestamps, has irregular 4-second cadence, or cannot be matched cleanly.

## What I Would Change

- I would not lead with dropping `unknown` unless you first confirm whether you are using only original files or mixing in processed files. The bundle readme says `unknown`/`unclassified` are part of processed events, but local EDA notes suggest they may also appear in originals. First measure actual labels from the files you load, then decide policy.
- I would exclude `irrigation` from the primary benchmark if the goal is indoor end-use discrimination, and report it separately as an auxiliary result. That keeps the main metric from being dominated by an easy class.
- I would convert raw pulses to a per-timestep volume first, then derive flow rate cleanly. With 4-second bins and `MeterResolution` in gallons/pulse, use a documented formula to get L/min:

```text
liters_per_bin = pulses * meter_resolution_gal_per_pulse * 3.78541
flow_l_min = liters_per_bin * (60 / 4)
```

- I would treat 128-step resampling as a model-input representation choice, not the only preserved data product. Keep original-length aligned sequences too, since duration may be predictive and useful later for masking or hybrid models.

## Recommended Order

1. Inventory labels and per-site counts from the chosen event source.
2. Normalize timestamps and validate event-to-QC alignment on a few sample sites.
3. Join `MeterResolution` and convert pulses to comparable physical units.
4. Extract raw event windows with QC diagnostics.
5. Decide final label policy: main benchmark classes, excluded classes, and separately reported classes.
6. Create the site-level split with a saved manifest and verify class presence in val/test.
7. Generate two sequence views: original-length and 128-step interpolated.
8. Emit a final dataset summary with class counts, split counts, dropped-event reasons, and representative duration statistics.

## Likely Deliverables

- A clean event table with one row per labeled event plus metadata and split assignment.
- A sequence dataset keyed by event ID with raw aligned traces and resampled 128-step traces.
- A reproducible site split manifest.
- A preprocessing report covering alignment failures, label exclusions, and split label coverage.

