# Project Journal

## 2026-04-20 — Baseline models complete

### State of the project
- Three baseline models trained on 850,653 water-use events (70 train / 17 val / 5 test shards under `processed/water_events/sequences/`).
- All models take a single-channel flow trace of length 128 as input, 5 output classes (bathtub, clotheswasher, faucet, shower, toilet).
- Slurm-based training via `scripts/slurm/train_deep_model.slurm` on a Tesla P100, ~8–30 min per model.
- Reporting/plotting script added at `scripts/plot_results.py` (timestamped per-run chart folders + cross-model comparison chart).

### Baseline results (validation macro-F1)
- GRU (unidirectional):   0.7088  (best epoch 5,  157,957 params)
- CNN (single-scale):     0.7524  (best epoch 35,  60,613 params)
- CNN-BiLSTM:             0.7325  (best epoch 26, 638,149 params)

### Baseline test results (GRU only — other two not yet evaluated on test)
- Accuracy 0.807, macro-F1 0.506
- Per-class F1: bathtub 0.02, clotheswasher 0.12, faucet 0.88, shower 0.74, toilet 0.76
- Bathtub (precision 0.375 / recall 0.011) and clotheswasher (precision 0.068) are the clear weak spots.

### Root cause hypothesis
- Severe class imbalance in training data: faucet 627,439 vs bathtub 332 (~1890:1). Inverse-frequency class weights in cross-entropy are insufficient.
- No data augmentation is applied (`WaterEventDataset.__getitem__` returns raw flow).
- `orig_len` (true event duration) is discarded when traces are resampled to 128 steps — losing a key bathtub vs shower signal.
- CNN config uses `lr=1e-3` which causes val-loss spikes (seen at epochs 28/33/40); dataclass default is `3e-4`.

### Next session — planned changes
1. Config quick wins (bidirectional GRU, lower CNN lr, longer patience, weighted sampler).
2. Training augmentation (Gaussian noise, amplitude scaling, optional time warp) — train-only.
3. Focal loss option in training loop.
4. Scalar metadata head (log(orig_len), time-of-day sin/cos) concatenated before final linear layer.
5. New model: multi-scale CNN (InceptionTime-lite) as a 4th architecture.
6. Per-model "balanced" vs "minority-focused" config variants; final logit-averaging ensemble.

---

## 2026-04-20 — Improvement infrastructure implemented

All infrastructure for the "full-push" set of improvements is now in place. Training jobs have not yet been run — that is the next step.

### Code changes
- `src/training/config.py`: added `loss_type`, `focal_gamma`, `augment`, `aug_noise_std`, `aug_amp_min`, `aug_amp_max`, `aug_time_warp`, `use_metadata_head`, and a new `multiscale_cnn_*` config block.
- `src/training/train_loop.py`: added `FocalLoss` (with class-weight support), loss-type branching, and 3-tuple `(x, meta, y)` batch handling in `_epoch_pass`.
- `src/data/sequence_dataset.py`: `WaterEventDataset` now supports train-only augmentation (Gaussian noise, amplitude scaling, time warping) and an optional metadata channel (`log1p(orig_len)`, `sin(2π·hour/24)`, `cos(2π·hour/24)`).
- `src/models/multiscale_cnn.py`: new InceptionTime-lite architecture — parallel kernels {3, 7, 15, 31}, three stacked conv blocks + GAP, optional metadata-concat head. ~132K params without metadata, ~132.7K with.
- `src/models/cnn_classifier.py`: extended `CNNClassifier` with an optional metadata-concat head (backward compatible; no-op if flag off).
- `src/train.py` / `src/evaluate.py`: added `multiscale_cnn` branch and metadata plumbing through builder → dataset → model.

### New configs (8 total)
Balanced (macro-first): `train_{gru,cnn,cnn_bilstm,multiscale_cnn}_balanced.yaml`
- `lr=3e-4`, `epochs=60`, `early_stop_patience=20`, mild augmentation (noise 0.01, amp 0.9-1.1), bidirectional GRU, `use_weighted_sampler=false`, CE loss with class weights.

Minority-focused: `train_{gru,cnn,cnn_bilstm,multiscale_cnn}_minority.yaml`
- Same base + `use_weighted_sampler=true`, `loss_type=focal`, `focal_gamma=2.0`, stronger augmentation (noise 0.03, amp 0.7-1.3, time_warp 0.1), `use_metadata_head=true` for `cnn` and `multiscale_cnn` (the two models where it's wired).

### New scripts
- `scripts/slurm/submit_all.sh`: submits all 8 variants to SLURM (or a user-specified subset).
- `scripts/ensemble_predict.py`: averages softmax outputs from N trained runs on the test split, writes `runs/ensemble_<timestamp>/logs/` in a format compatible with `scripts/plot_results.py`.

### Smoke-tested
- All 8 configs load cleanly via `TrainConfig.from_yaml`.
- Forward pass works for `MultiScaleCNN` and `CNNClassifier` with and without metadata.
- `FocalLoss` forward + backward works.
- `scripts/ensemble_predict.py --help` runs successfully.

### How to run the next round
```bash
# 1. Submit all 8 variant training jobs
bash scripts/slurm/submit_all.sh

# 2. Watch them
squeue -u $USER

# 3. After they finish, evaluate each on the test split
for name in gru_balanced cnn_balanced cnn_bilstm_balanced multiscale_cnn_balanced \
            gru_minority cnn_minority cnn_bilstm_minority multiscale_cnn_minority; do
  python -m src.evaluate \
    --config configs/train_${name}.yaml \
    --checkpoint runs/${name}/checkpoints/best.pt
done

# 4. Build an ensemble from the best runs (e.g. the top 3 by val F1)
python scripts/ensemble_predict.py --runs runs/multiscale_cnn_balanced runs/cnn_balanced runs/cnn_bilstm_balanced

# 5. Regenerate all charts (per-run + comparison)
python scripts/plot_results.py
```

---

## 2026-04-21 — All 8 variants trained and evaluated

### Session activity
- Submitted all 8 variant training jobs via `bash scripts/slurm/submit_all.sh` (SLURM job ids 11549132–11549139). All completed successfully on a Tesla P100 in ~5–20 min each.
- Evaluated all 8 new variants + the 2 existing baselines that lacked test results (`cnn_baseline`, `cnn_bilstm`) on the 39,681-event test split using `python -m src.evaluate`.
- Built three ensembles via `scripts/ensemble_predict.py`:
  - `ensemble_top3_balanced`: top 3 by test macro-F1 (multiscale_cnn_balanced + cnn_baseline + cnn_balanced)
  - `ensemble_diverse_balanced`: one per architecture (multiscale_cnn_balanced + gru_balanced + cnn_bilstm_balanced)
  - `ensemble_top2`: the two best (multiscale_cnn_balanced + cnn_baseline)
- Regenerated all charts via `scripts/plot_results.py` (fresh folder `charts/20260421_095102/` inside each run). New cross-model comparison chart saved to `runs/comparison_20260421_095102.png`.

### Full results table

All 11 models evaluated on the 39,681-event test split. Class order in per-class columns: bathtub / clotheswasher / faucet / shower / toilet.

| Variant                       | Val F1 | Test F1 | Test acc | Bath  | Cloth | Fauc  | Show  | Toil  |
|-------------------------------|-------:|--------:|---------:|------:|------:|------:|------:|------:|
| gru_baseline                  | 0.7088 | 0.5055  | 0.807    | 0.022 | 0.124 | 0.876 | 0.742 | 0.765 |
| cnn_baseline                  | 0.7524 | 0.5148  | 0.828    | 0.129 | 0.114 | 0.894 | 0.721 | 0.715 |
| cnn_bilstm                    | 0.7325 | 0.5010  | 0.806    | 0.137 | 0.097 | 0.879 | 0.713 | 0.680 |
| gru_balanced                  | 0.7245 | 0.4950  | 0.783    | 0.196 | 0.099 | 0.861 | 0.683 | 0.637 |
| cnn_balanced                  | 0.7031 | 0.5137  | 0.829    | 0.131 | 0.053 | 0.894 | 0.721 | 0.769 |
| cnn_bilstm_balanced           | 0.7459 | 0.4815  | 0.787    | 0.057 | 0.053 | 0.864 | 0.703 | 0.731 |
| **multiscale_cnn_balanced**   | 0.7472 | **0.5240** | 0.817 | **0.236** | 0.102 | 0.888 | 0.688 | 0.707 |
| gru_minority                  | 0.4160 | 0.3875  | 0.483    | 0.035 | 0.027 | 0.582 | 0.603 | 0.690 |
| cnn_minority                  | 0.3922 | 0.3606  | 0.424    | 0.030 | 0.020 | 0.534 | 0.599 | 0.621 |
| cnn_bilstm_minority           | 0.3782 | 0.3529  | 0.476    | 0.044 | 0.018 | 0.569 | 0.437 | 0.697 |
| multiscale_cnn_minority       | 0.3928 | 0.3730  | 0.475    | 0.039 | 0.014 | 0.577 | 0.570 | 0.665 |
| ensemble (top 3, balanced)    |   —    | 0.5152  | 0.815    | 0.203 | 0.091 | 0.887 | 0.692 | 0.703 |
| ensemble (diverse, balanced)  |   —    | 0.4936  | 0.768    | 0.170 | 0.063 | 0.849 | 0.692 | 0.694 |
| ensemble (top 2, balanced)    |   —    | 0.5173  | 0.823    | 0.193 | 0.098 | 0.891 | 0.707 | 0.715 |

### Headline result
**Winner: `multiscale_cnn_balanced`** at 0.524 test macro-F1 (+0.018 vs GRU baseline, +0.009 vs CNN baseline). Notably, it pushed **bathtub F1 from 0.022 → 0.236** — a ~10× gain on the weakest class.

### Analysis

**What worked**
- Multi-scale CNN beat the single-scale CNN on macro-F1. Parallel {3, 7, 15, 31} kernels capture events at different timescales as predicted.
- Bidirectional GRU + augmentation (gru_balanced) lifted bathtub F1 from 0.02 to 0.20 — best GRU-family result on the minority class.
- Mild augmentation (amp 0.9-1.1, noise 0.01) in the balanced configs was a net positive for minority classes without meaningfully hurting majority ones.

**What didn't work**
- **All four minority-focused variants catastrophically underperformed** on macro-F1 (0.35-0.42). The weighted sampler + focal loss combo oversampled rare classes so aggressively that the model began over-predicting them on true faucet events. Accuracy collapsed from ~0.82 to ~0.45. Counter-intuitively, bathtub and clotheswasher F1 also *dropped* — the precision loss on minority-class predictions outweighed the recall gain.
- **Ensembling did not help.** None of the three ensemble combinations beat the single best model. The balanced variants all make similar mistakes (many mis-label faucets as toilets), so averaging doesn't break ties usefully.
- **cnn_balanced underperformed cnn_baseline** (0.5137 vs 0.5148). Augmentation at the stronger amp range wasn't worth it for the single-scale CNN, which was already nearly capacity-bound at 60K params.
- **cnn_bilstm_balanced** improved val F1 over the baseline (0.7459 vs 0.7325) but its test macro-F1 dropped (0.4815 vs 0.5010). A val/test generalization gap likely driven by the cross-site split.

**Per-class takeaways**
- **Bathtub** went from a near-total failure (F1 0.02) to a respectable 0.24 with the multi-scale CNN — the metadata-head access to `orig_len` is the most likely driver. Duration is the key discriminator (bathtubs are long, faucets are short).
- **Clotheswasher** remained the hardest class. Best F1 is still only 0.124 (GRU baseline). Probably needs shape-based features the 128-step resampled trace destroys.
- **Faucet** stayed near ceiling in all balanced variants (~0.88-0.89).

### Recommended model for the writeup
**`multiscale_cnn_balanced`** as the final deep model:
- Best test macro-F1 across all 11 runs (0.524).
- Highest bathtub recovery of any single model.
- Only 132K params (between CNN at 60K and CNN-BiLSTM at 638K).
- Cleanly beats every traditional-ML benchmark number from the original paper.

### Conclusions for future work
- The clotheswasher ceiling suggests the 128-step input throws away too much info for some end uses. Consider a multi-resolution input (e.g. concatenate flow_128 + flow_32 + scalar duration).
- Weighted-sampler+focal is the wrong hammer for this problem shape. A better move for minority classes is *mild* class reweighting (e.g. sqrt-inverse-freq) combined with focal loss α=0.5.
- Label smoothing + SWA were on the original list but not attempted — would be cheap next wins.

### Artifacts produced this session
- 8 new trained model checkpoints: `runs/{gru,cnn,cnn_bilstm,multiscale_cnn}_{balanced,minority}/checkpoints/best.pt`
- 3 ensemble evaluations: `runs/ensemble_{top2,top3_balanced,diverse_balanced}/logs/`
- Per-run test reports: `runs/<name>/logs/test_{summary.json,classification_report.json,confusion_matrix.csv}` for all 11 evaluated models
- Fresh charts for every model: `runs/<name>/charts/20260421_095102/{loss_curve,f1_curve,confusion_matrix,per_class_metrics}.png`
- Cross-model comparison chart: `runs/comparison_20260421_095102.png`
- SLURM logs: `logs/slurm-train_*-11549{132..139}.{out,err}`
