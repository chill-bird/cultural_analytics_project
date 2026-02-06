#!/bin/bash
#SBATCH --job-name=lmske_keyframes
#SBATCH --partition=paula    
#SBATCH --array=0-237%16         
#SBATCH --gres=gpu:1      
#SBATCH --gpu-bind=single:1               
#SBATCH --cpus-per-task=8                
#SBATCH --mem=32G                        
#SBATCH --time=05:00:00   
#SBATCH --signal=B:SIGTERM@60 
#SBATCH --output=logs/keyframes_%A_%a.out
#SBATCH --error=logs/keyframes_%A_%a.err

# ----------------------------
# Load modules
# ----------------------------
module load Python/3.10
module load CUDA/11.8.0

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

VIDEO_DIR=/work/ow52opul-wochenschau_analysis/dat/video_data
FEATURE_DIR="$PWD/clip_features"
SCENE_DIR=/work/ow52opul-wochenschau_analysis/dat/scenes
OUT_DIR="$PWD/keyframes"

mkdir -p "$OUT_DIR"

FEATURE_FILES=("$FEATURE_DIR"/*_fixed.pkl)
FEATURE_FILE="${FEATURE_FILES[$SLURM_ARRAY_TASK_ID]}"

BASENAME=$(basename "$FEATURE_FILE" _fixed.pkl)

VIDEO_PATH="$VIDEO_DIR/${BASENAME}.mp4"
SCENE_PATH="$SCENE_DIR/${BASENAME}.mp4.scenes.txt"

echo "Processing $BASENAME"

# -------------------------
# Run keyframe extraction
# -------------------------
START=$(date +%s)
python run_keyframes.py \
  --video_file "$VIDEO_PATH" \
  --feature_file "$FEATURE_FILE" \
  --scene_file "$SCENE_PATH" \
  --output_dir "$OUT_DIR" \
  --video_name "$BASENAME"

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"