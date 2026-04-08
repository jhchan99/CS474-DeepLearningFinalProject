# Water-Use Dataset Exploration

## Inventory
| Dataset | Files | Rows | Sites |
| --- | --- | --- | --- |
| sites_metadata | 1 | 31 | 31 |
| log_files | 31 | 264 | 31 |
| original_events | 31 | 871353 | 31 |
| processed_events | 31 | 871353 | 31 |
| raw_qc_data | 31 | 43869547 | 31 |

## Event Highlights
- Original event labels: faucet, toilet, unclassified, shower, irrigation, clotheswasher, bathtub, unknown
- Processed event labels: unclassified, faucet, toilet, shower, irrigation, clotheswasher, bathtub, unknown
- Dominant processed label: `unclassified` (78.9% of processed events).
- Median processed duration: 0.07 minutes.
- Median processed volume: 0.03 gallons.

## Raw Sequence Highlights
- Total raw rows scanned: 43,869,547.
- Mean site zero-pulse fraction: 90.0%.
- Raw coverage range: 44.9 to 508.0 days.
- Total estimated missing 4-second steps: 130,016,363.

## Metadata Join Highlights
- Highest event-count site: 3 with 388,545 processed events.
- Highest total-pulse site: 5 with 8,628,980 pulses.

## Paper Table 2 Site Reference
- Added as a supplemental site lookup keyed by `Site` from the paper-provided table.
- Kept separate from `sites.csv` columns because some values differ between the paper table and HydroShare metadata.
| Site | QCRecordWeeks_Paper | Occupants_Paper | IrrigableArea_m2_Paper | BuildingArea_m2_Paper | IrrigationMode_Paper | PulseResolution_L_per_pulse_Paper | AnnualAverageWaterUse_m3_Paper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 21.6 | 2 | 643 | 140 | Sprinkler system | 0.13 | 397.9 |
| 3 | 22.5 | 2 | 1408 | 138 | Hose | 0.03 | 234.9 |
| 4 | 9.3 | 4 | 1015 | 136 | Sprinkler system | 0.03 | 647.7 |
| 5 | 16.4 | 2 | 3118 | 169 | Sprinkler system | 0.13 | 1786 |
| 6 | 6.2 | 3 | 294 | 101 | Hose | 0.03 | 96.2 |
| 7 | 16.8 | 3 | 241 | 104 | Hose | 0.13 | 55.2 |
| 8 | 6.4 | 2 | 1789 | 160 | Hose | 0.13 | 720.6 |
| 9 | 9.3 | 2 | 509 | 173 | Sprinkler system | 0.13 | 602.4 |
| 10 | 6.2 | 2 | 824 | 102 | Hose | 0.13 | 149 |
| 11 | 12.4 | 4 | 827 | 136 | Sprinkler system | 0.13 | 401.3 |
| 12 | 9.9 | 2 | 1744 | 156 | Sprinkler system | 0.13 | 181.2 |
| 13 | 7.8 | 2 | 742 | 239 | Sprinkler system | 0.13 | 1507.5 |
| 14 | 10.4 | 2 | 2005 | 315 | Sprinkler system | 0.13 | 1099.8 |
| 15 | 9 | 6 | 405 | 171 | Sprinkler system | 0.13 | 392.2 |
| 16 | 7.4 | 3 | 1162 | 151 | Hose | 0.03 | 247.4 |
| 17 | 8.5 | 3 | 1451 | 92 | Hose | 0.03 | 341 |
| 18 | 8.7 | 1 | 410 | 74 | Hose | 0.03 | 233.5 |
| 19 | 23.1 | 5 | 982 | 128 | Sprinkler system | 0.16 | 854.1 |
| 20 | 5.2 | 4 | 1202 | 177 | Sprinkler system | 0.16 | 942.2 |
| 21 | 6.8 | 6 | N/A | N/A | Sprinkler system | 0.16 | N/A |
| 22 | 8 | 7 | N/A | N/A | Sprinkler system | 0.16 | N/A |
| 23 | 5.7 | 6 | 1108 | 144 | Sprinkler system | 0.16 | 809.2 |
| 24 | 8.1 | 8 | 1276 | 279 | Sprinkler system | 0.16 | 1308 |
| 25 | 7.4 | 6 | 914 | 282 | Sprinkler system | 0.16 | 614.4 |
| 26 | 6.7 | 6 | 3592 | 117 | Sprinkler system | 0.1 | 644 |
| 27 | 6.7 | 7 | 3842 | 299 | Sprinkler system | 0.1 | 2248.6 |
| 28 | 4.8 | 6 | 1846 | 337 | Sprinkler system | 0.16 | 1747.6 |
| 29 | 4.8 | 3 | 700 | 133 | Sprinkler system | 0.16 | 573.3 |
| 30 | 5 | 6 | 1250 | 137 | Sprinkler system | 0.16 | 716.2 |
| 31 | 4.6 | 3 | 827 | 154 | Sprinkler system | 0.13 | 695.7 |
| 32 | 4.6 | 2 | 862 | 104 | Sprinkler system | 0.03 | 730.1 |

Plots are saved under `data_exploration/plots/` and summary tables under `data_exploration/tables/`.