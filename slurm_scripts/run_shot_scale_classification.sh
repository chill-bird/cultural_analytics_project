#!/bin/bash
#SBATCH --job-name=classify_shot_scales
#SBATCH --partition=paula
#SBATCH --array=0-1545
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:10:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/classify_shot_scale/%x_%A_%a.out
#SBATCH --error=logs/classify_shot_scale/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Modules
# -------------------------
module purge

# Activate Python environment
source "/work/ow52opul-wochenschau_analysis/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FF_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------
# Sanity checks
# -------------------------
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPUs visible:"

# Fail immediately if CUDA is broken
python - << 'EOF'
import torch, sys
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — ABORTING"
print("CUDA OK:", torch.cuda.get_device_name(0))
EOF

# -------------------------
# Run shot scale classification
# -------------------------
START=$(date +%s)

python -m src.keyframes.classify_shot_scale

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"
