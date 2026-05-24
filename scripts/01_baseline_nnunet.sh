#!/bin/bash
#SBATCH --job-name=baseline_nnunet_f${FOLD:-0}
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
# EXPERIMENT 0: Vanilla nnU-Net (reference).
# Uses default plans (with z-score normalization).
# Expected fold 0: ~0.825 (from prior runs in results/lipo_nnunet/).
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

FOLD=${FOLD:-0}
DATASET_ID=500
CONFIG=3d_fullres

echo "[exp 0: vanilla] fold=$FOLD start $(date)"
LATEST="$nnUNet_results/Dataset500_Lipo/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_latest.pth"
CONT_FLAG=""
if [ -f "$LATEST" ]; then
    echo "[exp 0: vanilla] resuming from $LATEST"
    CONT_FLAG="--c"
fi
nnUNetv2_train $DATASET_ID $CONFIG $FOLD --npz $CONT_FLAG
echo "[exp 0: vanilla] fold=$FOLD end $(date)"
