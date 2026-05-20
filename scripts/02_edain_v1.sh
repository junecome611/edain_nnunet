#!/bin/bash
#SBATCH --job-name=edain_v1_f${FOLD:-0}
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# EXPERIMENT 1: EDAIN v1 (h1 + h2 + h3, no power transform).
# Input: raw MRI (nnUNetPlans_raw, NoNormalization).
# EDAIN replaces the upstream z-score with a learnable per-image normalization.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0
source ~/nnunet_env/bin/activate

export nnUNet_raw=${nnUNet_raw:-$HOME/nnUNet_data/raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnUNet_data/results}
export nnUNet_n_proc_DA=0
export nnUNet_compile=F

REPO_ROOT="$SLURM_SUBMIT_DIR/.."
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FOLD=${FOLD:-0}
DATASET_ID=500
CONFIG=3d_fullres

# Path to precomputed stats (produced by 00_setup_data.sh)
export EDAIN_V1_STATS_JSON="$REPO_ROOT/edain_v1_stats/edain_v1_stats_fold${FOLD}.json"

# h4 OFF (this is EDAIN v1 without power transform)
export EDAIN_V1_USE_POWER=0
# Use percentile rescale to [0,1] before EDAIN (recommended for raw MRI)
export EDAIN_V1_RESCALE_P2P98=1

# Per-sublayer LR multipliers (defaults from paper grid search)
export EDAIN_V1_LR_ALPHA=10
export EDAIN_V1_LR_BETA=10
export EDAIN_V1_LR_SHIFT=1
export EDAIN_V1_LR_SCALE=1

echo "[exp 1: EDAIN v1] fold=$FOLD start $(date)"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
    -tr nnUNetTrainerEDAINv1 \
    -p nnUNetPlans_raw \
    --npz
echo "[exp 1: EDAIN v1] fold=$FOLD end $(date)"
