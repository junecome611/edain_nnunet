#!/bin/bash
#SBATCH --job-name=nyul_popnyul_f${FOLD:-0}
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
# EXPERIMENT 4: Nyul-popnyul (our v2 spline + hypernet, classical Nyul anchor).
# Uses default z-score preprocessing. Spline starts as population Nyul mapping.
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

# Nyul subclass with anchor_type=population_nyul, outlier_clip=none baked in.
echo "[exp 4: Nyul popnyul] fold=$FOLD start $(date)"
LATEST="$nnUNet_results/Dataset500_Lipo/nnUNetTrainerNyulPopnyul__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_latest.pth"
CONT_FLAG=""
if [ -f "$LATEST" ]; then
    echo "[exp 4: Nyul popnyul] resuming from $LATEST"
    CONT_FLAG="--c"
fi
nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
    -tr nnUNetTrainerNyulPopnyul \
    --npz $CONT_FLAG
echo "[exp 4: Nyul popnyul] fold=$FOLD end $(date)"
