#!/bin/bash
#SBATCH --job-name=detect_blackouts
#SBATCH --partition=paula
#SBATCH --array=0-237%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:07:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Environment
# -------------------------
module purge
module load FFmpeg

# Activate Python environment
source "$PWD/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FF_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------
# Sanity checks
# -------------------------
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node:       $(hostname)"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"

python - << 'EOF'
import cv2, imageio, os, sys
print("OpenCV:", cv2.__version__)
print("imageio OK")
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
EOF

# -------------------------
# Run blackout detection
# -------------------------
START=$(date +%s)

python -m src.split.find_blackout_segments

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"
