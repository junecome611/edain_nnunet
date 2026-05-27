#!/bin/bash
#SBATCH --job-name=edain_v1_power_f${FOLD:-0}
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# Only gpu005/gpu006 are known-good; other nodes have broken GPUs.
#SBATCH --nodelist=gpu[005-006]
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# EXPERIMENT 2: EDAIN v1 + Power Transform (h1 + h2 + h3 + h4).
# Adds the Yeo-Johnson power transform on top of EDAIN v1.
# Input: raw MRI (nnUNetPlans_raw).
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

# Ensure our trainers are reachable by nnUNetv2_train
python -m tools.register_trainers >/dev/null

FOLD=${FOLD:-0}
DATASET_ID=500
CONFIG=3d_fullres

export EDAIN_V1_STATS_JSON="$REPO_ROOT/edain_v1_stats/edain_v1_stats_fold${FOLD}.json"

# h4 ON (this is the key difference from experiment 2)
export EDAIN_V1_USE_POWER=1

export EDAIN_V1_LR_ALPHA=10
export EDAIN_V1_LR_BETA=10
export EDAIN_V1_LR_SHIFT=1
export EDAIN_V1_LR_SCALE=1
export EDAIN_V1_LR_POWER=10

echo "[exp 2: EDAIN v1 + power] fold=$FOLD start $(date)"
LATEST="$nnUNet_results/Dataset500_Lipo/nnUNetTrainerEDAINv1Power__nnUNetPlans_raw__${CONFIG}/fold_${FOLD}/checkpoint_latest.pth"
CONT_FLAG=""
if [ -f "$LATEST" ]; then
    echo "[exp 2: EDAIN v1 + power] resuming from $LATEST"
    CONT_FLAG="--c"
fi
nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
    -tr nnUNetTrainerEDAINv1Power \
    -p nnUNetPlans_raw \
    --npz $CONT_FLAG
echo "[exp 2: EDAIN v1 + power] fold=$FOLD end $(date)"
