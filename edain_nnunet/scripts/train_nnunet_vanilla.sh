#!/bin/bash
#SBATCH --job-name=nnunet_vanilla_f${FOLD:-0}
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu[005-006]
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ===========================================================================
# Baseline: vanilla nnU-Net v2 on Lipo, 3d_fullres, fold $FOLD.
# This is the SAME as code/run_nnunet_baseline.sh but driven per-fold so we
# can run all configurations of the EDAIN ablation as siblings under
# edain_nnunet/scripts/.
# ===========================================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0
source ~/nnunet_env/bin/activate

export nnUNet_raw=/trinity/home/r112643/nnUNet_data/raw
export nnUNet_preprocessed=/trinity/home/r112643/nnUNet_data/preprocessed
export nnUNet_results=/trinity/home/r112643/nnUNet_data/results
export nnUNet_n_proc_DA=0
export nnUNet_compile=F

FOLD=${FOLD:-0}
DATASET_ID=500
CONFIG=3d_fullres

echo "[vanilla nnU-Net] fold=$FOLD | start $(date)"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD --npz
echo "[vanilla nnU-Net] fold=$FOLD | end $(date)"
