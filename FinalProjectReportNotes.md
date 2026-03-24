# Final Project Report Notes

## Assignment Requirements

### Submission

- Submit a PDF writeup.
- Maximum length: 3 pages total.
- The writeup must include:
  - A brief description of the problem, data, technical approach, and results.
  - A time log showing total time spent, broken down by day.

### Length Guidance

- Project description, data, approach, and results: no more than 2 pages.
- Time log: no more than 1 page.

### Grading

- `20%`: number of hours spent
- `5%`: report quality

### Time Log Rules

- Time must be documented daily with a short description of the work done.
- Undocumented time does not count.
- No more than `5` hours of research and reading may count.
- No more than `10` hours of prep work may count.
- At least `20` hours must involve designing, building, debugging, testing deep learning models, analyzing results, or experimenting.
- Model training runtime does not count toward hours.
- More than `30` hours does not earn extra credit.

### Project Rules

- The project must include training or finetuning a model.
- Using a pretrained model only for inference is not enough.
- The project will be graded more on effort than results.
- The final report should be polished, clear, and high quality.

### Report Content

- The problem being solved
- Exploratory data analysis
- Technical approach
- Results

### Questions the Report Should Answer

#### Dataset

- Where did it come from?
- Who published it?
- Why does this data matter?

#### Problem Framing

- Is it classification or regression?
- Is it supervised or unsupervised?
- What background knowledge is relevant?
- What prior approaches exist?

#### Data Exploration

- What does the dataset include?
- What patterns are visible before modeling?
- What visualizations are relevant?

#### Technical Approach

- What model is used?
- What training and inference approach is used?
- How is the dataset partitioned?
- How many parameters does the model have?
- What optimizer is used?
- Why was that topology chosen?
- Were pretrained weights used?

#### Results and Analysis

- What was the final performance on the private split?
- Did the model overfit?
- Did the design iterate over time?
- Was progress made on the original problem?

## Our Project Direction

### Problem

We are planning a supervised deep learning project for residential water end-use classification. The target task is to classify household water-use events into categories such as `faucet`, `toilet`, `shower`, `irrigation`, `clotheswasher`, and `bathtub`, using high temporal resolution water meter data.

### Data

- Main dataset: Cache County high-resolution residential water-use dataset from HydroShare.
- Dataset citation currently in `README.md`:
  - Bastidas Pacheco, C. J., N. Atallah, J. S. Horsburgh (2023). *High Resolution Residential Water Use Data in Cache County, Utah, USA*, HydroShare, [https://doi.org/10.4211/hs.0b72cddfc51c45b188e0e6cd8927227e](https://doi.org/10.4211/hs.0b72cddfc51c45b188e0e6cd8927227e)
- Supporting paper:
  - `bastidas-pacheco-et-al-2022-variability-in-consumption-and-end-uses-of-water-for-residential-users-in-logan-and.pdf`
- Data-collection / sensing-system paper:
  - `sensors-20-03655-v2.pdf`

### Why This Dataset Matters

- It contains high-resolution residential water-use measurements at `4 s` resolution.
- It includes labeled end-use events.
- It is a realistic, domain-specific dataset rather than a generic benchmark.
- The problem has practical relevance for water demand analysis, conservation, and smart-meter analytics.

## What We Have Done So Far

### Project Framing

- Confirmed that the project must involve actual model training, so we are using a deep learning approach rather than inference-only methods.
- Chosen a classification framing centered on water end-use event classification.
- Decided to use a hybrid workflow:
  - first do event-level exploration and baseline models
  - then build a sequence model using raw `3_QC_Data`

### Dataset Understanding

- Identified the key folders in the HydroShare bundle:
  - `1_SecondaryData`
  - `2_LogFiles`
  - `3_QC_Data`
  - `4_EventFilesOriginal`
  - `5_EnventFiles_Processed`
- Confirmed from the dataset readme that:
  - `3_QC_Data` contains raw `4 s` pulse measurements
  - event files contain labeled water-use events
  - site metadata includes `MeterResolution`, `MeterBrand`, `MeterSize`, and household characteristics

### Literature / Context

- Reviewed the case-study paper to ground the problem context and final citation strategy.
- Reviewed the sensing-system paper to understand how the data were collected and what assumptions to make about:
  - `4 s` sampling
  - meter accuracy
  - raw pulses
  - meter metadata and calibration context

### Technical Decisions Made

- Main framework: `PyTorch`
- Workflow style: standard Python files only
- We are not using Jupyter notebooks.
- We are not using a heavyweight orchestration framework like Airflow or Kedro.
- We are using a modular, script-based pipeline.

### Planned Software Stack

#### Core

- `python`
- `pytorch`
- `numpy`
- `pandas`
- `scikit-learn`
- `torchmetrics`
- `pyyaml`
- `tqdm`

#### Visualization / Reporting

- `matplotlib`
- `seaborn`
- `tensorboard`

#### Optional

- `xgboost` or `lightgbm` for stronger tabular baselines
- `imbalanced-learn` if class imbalance becomes a major issue

### Pipeline Strategy

- Use plain Python modules and scripts.
- Use `pandas` and `numpy` for ingestion and preprocessing.
- Use `scikit-learn` utilities for splits, encoders, and baseline preprocessing.
- Use PyTorch `Dataset` and `DataLoader` classes for training.
- Use config-driven runs with YAML files.
- Keep everything reproducible and SLURM-friendly.

## Current Project Plan

### 1. Data Exploration

- Inventory all available data sources.
- Measure class balance, site coverage, event duration, volume, and flow-rate distributions.
- Check for duplicate rows, missing timestamps, zero-duration events, logger gaps, and leakage risk across sites.

### 2. Preprocessing Pipeline

- Build a reproducible preprocessing script.
- Generate:
  - event-level tabular data for baselines
  - raw sequence data aligned with labels for the CNN + BiLSTM
- Use `MeterResolution` to preserve both pulse-based and gallon-based representations.
- Decide how to handle `unknown` and `unclassified` labels.

### 3. Dataset and DataLoader

- Implement dataset classes for:
  - event-level baseline data
  - variable-length sequence data
- Add padding, masking, and collate logic.
- Split by site to avoid train/test leakage.

### 4. Baseline Models

- Majority-class baseline
- Logistic regression
- Random forest
- Gradient boosting or XGBoost if needed
- Possibly a small MLP baseline

### 5. CNN + BiLSTM Model

- Use raw `4 s` sequences as the main neural input.
- Use 1D CNN layers for local temporal patterns.
- Use BiLSTM layers for longer temporal structure.
- Add a classification head for event labels.
- Consider optional side features or metadata.

### 6. Training and Validation

- Use held-out sites for validation and testing.
- Add early stopping, checkpoints, learning-rate scheduling, and deterministic seeds.
- Track per-class metrics and macro F1.

### 7. Evaluation and Analysis

- Compare baselines against the CNN + BiLSTM.
- Use confusion matrices and per-class precision/recall/F1.
- Analyze errors by class, site, event duration, and water-use characteristics.

### 8. HPC Setup and Tracking

- Use SLURM-friendly scripts.
- Log configs, metrics, checkpoints, and split manifests.
- Use local logging plus TensorBoard first.

## Planned Project Structure

- `src/data/`
- `src/models/`
- `src/training/`
- `src/evaluation/`
- `src/train.py`
- `src/evaluate.py`
- `configs/`
- `scripts/`
- `reports/`
- `runs/`

## What We Need To Do Next

### Immediate Next Steps

1. Create the actual project skeleton and dependency file.
2. Implement data exploration scripts for event files and site metadata.
3. Quantify the real label set and class imbalance.
4. Decide the exact split strategy by site.
5. Build the first preprocessing script that outputs clean event-level training data.

### After That

1. Implement baseline models and establish reference metrics.
2. Build the sequence dataset from raw `3_QC_Data`.
3. Implement the CNN + BiLSTM.
4. Run training experiments and compare results.
5. Draft the final PDF report and maintain the daily time log throughout.

## Final Report Notes

### Likely Report Structure

1. Problem and motivation
2. Dataset and source papers
3. Exploratory data analysis
4. Technical approach
5. Results and discussion
6. Time log appendix or final page

### Important Things To Remember

- The writeup should stay concise: `1-2` pages for the project itself.
- The time log must be maintained daily.
- Hours spent reading papers are capped.
- Hours spent waiting on training do not count.
- We should emphasize both technical effort and iteration, not just final accuracy.

