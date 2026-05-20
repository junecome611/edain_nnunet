#!/bin/bash
#SBATCH --job-name=nyul_identity_f${FOLD:-0}
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# EXPERIMENT 3: Nyul-identity (our v2 spline + hypernet).
# Uses default z-score preprocessing. Spline starts as identity.
# Prior best on Lipo fold 0 (Path AB): 0.8033.
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

# Nyul subclass with anchor_type=identity, outlier_clip=percentile baked in.
echo "[exp 3: Nyul identity] fold=$FOLD start $(date)"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
    -tr nnUNetTrainerNyulIdentity \
    --npz
echo "[exp 3: Nyul identity] fold=$FOLD end $(date)"
