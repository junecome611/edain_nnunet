#!/bin/bash
#SBATCH --job-name=nyul_popnyul_val_f${FOLD:-0}
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# Only gpu005/gpu006 are known-good; other nodes have broken GPUs.
#SBATCH --nodelist=gpu[005-006]
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# EXPERIMENT 4 (validation-only): Nyul-popnyul.
#
# Re-runs `perform_actual_validation` from the existing checkpoint_final.pth
# WITHOUT retraining. Picks up the three inference-path fixes from 4d445f4.
#
# REQUIRES
#     $nnUNet_results/Dataset500_Lipo/nnUNetTrainerNyulPopnyul__nnUNetPlans__3d_fullres/
#         fold_<FOLD>/checkpoint_final.pth     (from the previous training run)
#
# OUTPUTS
#     Overwrites $output_folder/validation/Lipo-*.nii.gz and summary.json.
#     Back the old (buggy) numbers up first if you want to compare side-by-side.
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

CKPT="$nnUNet_results/Dataset500_Lipo/nnUNetTrainerNyulPopnyul__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_final.pth"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found — was training finished?"
    exit 1
fi
echo "[exp 4: Nyul popnyul, val-only] fold=$FOLD using $CKPT"
echo "[exp 4: Nyul popnyul, val-only] start $(date)"

nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
    -tr nnUNetTrainerNyulPopnyul \
    --val --npz

echo "[exp 4: Nyul popnyul, val-only] fold=$FOLD end $(date)"
