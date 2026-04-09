#!/bin/bash
#
# Train a water end-use deep learning model on a GPU (or fast CPU) node.
#
# Submit from the project root:
#   sbatch scripts/slurm/train_deep_model.slurm --config configs/train_gru.yaml
#
# Quick smoke-test (test QOS, CPU only):
#   sbatch --qos=test --time=00:30:00 --gres='' \
#          scripts/slurm/train_deep_model.slurm --config configs/train_gru.yaml
#
# To pass --config and other python flags, set CONFIG and EXTRA_ARGS:
#   CONFIG=configs/train_cnn.yaml sbatch scripts/slurm/train_deep_model.slurm
#

#SBATCH --job-name=dl-water-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jamcha99/DeepLearningFinalProject/logs/slurm-%x-%j.out
#SBATCH --error=/home/jamcha99/DeepLearningFinalProject/logs/slurm-%x-%j.err

# Uncomment to target a specific partition or account:
# #SBATCH --partition=YOUR_GPU_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# Optional email notifications:
# #SBATCH --mail-user=you@byu.edu
# #SBATCH --mail-type=END,FAIL

set -euo pipefail

# ── Resolve project root (this file lives in scripts/slurm/) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

mkdir -p logs

# ── Config ───────────────────────────────────────────────────────────────────
# Default: gru_baseline. Override by exporting CONFIG before sbatch.
CONFIG="${CONFIG:-configs/train_gru.yaml}"

# ── Thread caps ──────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

# ── Python environment ────────────────────────────────────────────────────────
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate dl-water

# ── GPU check ────────────────────────────────────────────────────────────────
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
           [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

# ── Train ─────────────────────────────────────────────────────────────────────
echo "=== Starting training: CONFIG=${CONFIG} ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME:-localhost}  Started: $(date -Is)"

python -m src.train --config "${CONFIG}" ${EXTRA_ARGS:-}

echo "=== Training complete: $(date -Is) ==="
