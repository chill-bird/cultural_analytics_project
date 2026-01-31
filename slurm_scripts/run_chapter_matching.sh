#!/bin/bash
#SBATCH --job-name=match_chapters
#SBATCH --partition=paula
#SBATCH --array=0-189
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/match_chapters/%x_%A_%a.out
#SBATCH --error=logs/match_chapters/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Modules
# -------------------------
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

# Activate Python environment
source "/work/ow52opul-wochenschau_analysis/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------
# Temporary offload dirs
# -------------------------
OFFLOAD_DIR="/tmp/$USER/model_offload_$SLURM_JOB_ID"
mkdir -p "$OFFLOAD_DIR"
export MODEL_OFFLOAD_DIR="$OFFLOAD_DIR"

# -------------------------
# Sanity checks
# -------------------------
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:       $(hostname)"
echo "GPUs visible:"
nvidia-smi

# Fail immediately if CUDA is broken
python - << 'EOF'
import torch, sys
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — ABORTING"
print("CUDA OK:", torch.cuda.get_device_name(0))
print("GPU 0:", torch.cuda.get_device_name(0))
print("GPU 1:", torch.cuda.get_device_name(1))
EOF

# -------------------------
# Run matching
# -------------------------
START=$(date +%s)

python -m src.match_chapters.match_chapters

# -------------------------
# Remove offload directory
# -------------------------
rm -rf "$OFFLOAD_DIR"

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"
