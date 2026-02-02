#!/bin/bash
#SBATCH --job-name=get_shot_boundaries_transnetv2   
#SBATCH --partition=paula    
#SBATCH --array=0-237%16         
#SBATCH --gres=gpu:1      
#SBATCH --gpu-bind=single:1               
#SBATCH --cpus-per-task=8                
#SBATCH --mem=32G                        
#SBATCH --time=00:10:00   
#SBATCH --signal=B:SIGTERM@60 
#SBATCH --output=logs/transnetv2_%j.out           
#SBATCH --error=logs/transnetv2_%j.err 

# ----------------------------
# Load modules
# ----------------------------

module purge
module load FFmpeg
module load CUDA/11.8.0
module load cuDNN/8.6.0.163

# Activate Python environment
source "$PWD/.venv/bin/activate"

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

python - <<'EOF'
import torch
import tensorflow as tf
print("Torch device:", "cuda" if torch.cuda.is_available() else "cpu")
print("TF devices:", tf.config.list_physical_devices())
EOF

VIDEO_DIR="/work/ow52opul-wochenschau_analysis/dat/video_data"
VIDEOS=("$VIDEO_DIR"/*.mp4)
VIDEO="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"

echo "Processing: $VIDEO"

# -------------------------
# Run matching
# -------------------------
START=$(date +%s)

python /work/xb27qenu-ca_lmske/TransNetV2/inference/transnetv2.py "$VIDEO" --visualize

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"