# Time alignment spot check

Policy: parse event `StartTime`/`EndTime` as naive by stripping `Z` (matches QC wall clock in bundle).
Slices are **inclusive** on QC `Time`.

## Site 2 (first 5 events)

[
  {
    "event_index": 0,
    "start_naive": "2019-04-01 14:44:15",
    "end_naive": "2019-04-01 14:44:23",
    "n_qc_rows": 3,
    "duration_min": 0.2
  },
  {
    "event_index": 1,
    "start_naive": "2019-04-01 15:35:35",
    "end_naive": "2019-04-01 15:36:23",
    "n_qc_rows": 13,
    "duration_min": 0.866666667
  },
  {
    "event_index": 2,
    "start_naive": "2019-04-01 15:44:11",
    "end_naive": "2019-04-01 15:46:43",
    "n_qc_rows": 39,
    "duration_min": 2.6
  },
  {
    "event_index": 3,
    "start_naive": "2019-04-01 15:46:59",
    "end_naive": "2019-04-01 15:47:15",
    "n_qc_rows": 5,
    "duration_min": 0.333333333
  },
  {
    "event_index": 4,
    "start_naive": "2019-04-01 15:47:55",
    "end_naive": "2019-04-01 15:47:55",
    "n_qc_rows": 1,
    "duration_min": 0.066666667
  }
]