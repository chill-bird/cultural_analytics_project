#!/bin/bash
#SBATCH --job-name=narrative_analysis
#SBATCH --array=0-1545
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=0:10:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Modules
# -------------------------
module purge

# Activate Python environment
source /work/xb27qenu-ca_narrative_detection/.venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export FF_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------
# Sanity checks
# -------------------------
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"

EPISODE_ID=$SLURM_ARRAY_TASK_ID

# -------------------------
# Run narrative detection
# -------------------------
START=$(date +%s)

python detect_narratives.py --episode $EPISODE_ID

END=$(date +%s)
echo "Total runtime: $((END - START)) seconds"


