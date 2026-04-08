# Label policy (implementation)

- **Source**: `4_EventFilesOriginal` / `LabelledEvents_site_*.csv` only.
- **Dropped** (not in `events_clean.csv`): `unknown`, `unclassified` (case-insensitive).
- **Benchmark task** (`events_benchmark.csv`): same as clean, plus **exclude** `irrigation` from the primary multiclass label set; irrigation rows are kept separately in `events_irrigation_only.csv`.
- Encoding for the benchmark task is in `label_encoding_benchmark.json` (all classes present in full benchmark file).
