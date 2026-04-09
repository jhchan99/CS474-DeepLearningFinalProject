#!/bin/bash
#
# Build the full water-event dataset on a compute node (BYU Slurm / similar).
# Submit from the project root, or from anywhere:
#   sbatch /path/to/DeepLearningFinalProject/scripts/slurm/build_water_event_dataset.slurm
#
# For quick workflow tests, use the cluster "test" QOS (short walltime / fast queue), e.g.:
#   sbatch --qos=test --time=00:30:00 scripts/slurm/build_water_event_dataset.slurm
#

#SBATCH --job-name=water-events-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jamcha99/DeepLearningFinalProject/logs/slurm-%x-%j.out
#SBATCH --error=/home/jamcha99/DeepLearningFinalProject/logs/slurm-%x-%j.err

# Uncomment if your site uses a non-default partition or account:
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT

# Optional notifications (uncomment and set your email):
# #SBATCH --mail-user=you@byu.edu
# #SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT="/home/jamcha99/DeepLearningFinalProject"
cd "${ROOT}"
mkdir -p "${ROOT}/logs"

# OpenMP / BLAS thread caps: match allocated CPUs (Slurm sets SLURM_CPUS_PER_TASK with cpus-per-task)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate dl-water

echo "Starting dataset build at $(date -Is)"
echo "Host: ${SLURMD_NODENAME:-unknown}  Job: ${SLURM_JOB_ID:-unknown}"
echo "Working directory: ${ROOT}"
echo "Python: $(which python)"
python -u "${ROOT}/scripts/build_water_event_dataset.py" --seed 42

echo "Finished at $(date -Is)"
