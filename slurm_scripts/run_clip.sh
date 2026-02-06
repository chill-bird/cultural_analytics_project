#!/bin/bash
#SBATCH --job-name=clip_features   
#SBATCH --partition=paula    
#SBATCH --array=0-237%16           
#SBATCH --gres=gpu:1      
#SBATCH --gpu-bind=single:1               
#SBATCH --cpus-per-task=8                
#SBATCH --mem=32G                        
#SBATCH --time=00:10:00   
#SBATCH --signal=B:SIGTERM@60 
#SBATCH --output=logs/clip_%j.out           
#SBATCH --error=logs/clip_%j.err      

# ----------------------------
# Load modules
# ----------------------------
module purge
module load Python/3.10
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

VIDEO_DIR=/work/ow52opul-wochenschau_analysis/dat/video_data
VIDEOS=("$VIDEO_DIR"/*.mp4)

VIDEO="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"

# -------------------------
# Run CLIP
# -------------------------
START=$(date +%s)

python /work/xb27qenu-ca_lmske/extract_clip_features.py "$VIDEO" \
    --output_dir clip_features \
    --fps 1.0 \
    --batch_size 32

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"
