#!/bin/bash
#SBATCH --job-name=transcribe_video_data
#SBATCH --partition=paula
#SBATCH --array=0-229%16
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Environment
# -------------------------
module purge
module load FFmpeg

# Environment must be existent before starting the job
source ~/whisper_env/bin/activate

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
nvidia-smi

# Fail immediately if CUDA is broken
python - << 'EOF'
import torch, sys
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — ABORTING"
print("CUDA OK:", torch.cuda.get_device_name(0))
EOF

# -------------------------
# Run transcription
# -------------------------
START=$(date +%s)

python -m src.transcribe.transcribe

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"
