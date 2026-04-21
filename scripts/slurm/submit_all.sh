#!/bin/bash
#
# Submit all 8 variant training jobs (balanced + minority-focused for each of
# the 4 models) to SLURM.
#
# Usage:
#   bash scripts/slurm/submit_all.sh
#
# To submit only a subset, pass config paths as arguments:
#   bash scripts/slurm/submit_all.sh configs/train_cnn_balanced.yaml configs/train_gru_minority.yaml
#
# The script prints the job id for each submission so you can track them later
# with ``squeue -u $USER``.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEFAULT_CONFIGS=(
  configs/train_gru_balanced.yaml
  configs/train_cnn_balanced.yaml
  configs/train_cnn_bilstm_balanced.yaml
  configs/train_multiscale_cnn_balanced.yaml
  configs/train_gru_minority.yaml
  configs/train_cnn_minority.yaml
  configs/train_cnn_bilstm_minority.yaml
  configs/train_multiscale_cnn_minority.yaml
)

if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "Submitting ${#CONFIGS[@]} training job(s)..."
echo ""

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "  [skip] ${cfg} — file not found"
    continue
  fi
  name="$(basename "${cfg}" .yaml)"
  jobinfo=$(CONFIG="${cfg}" sbatch --job-name="${name}" scripts/slurm/train_deep_model.slurm)
  echo "  ${cfg}  →  ${jobinfo}"
done

echo ""
echo "Use 'squeue -u \$USER' to monitor progress."
echo "Logs stream to logs/slurm-<job_name>-<jobid>.out"
