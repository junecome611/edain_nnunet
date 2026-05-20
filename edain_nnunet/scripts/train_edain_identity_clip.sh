#!/bin/bash
#SBATCH --job-name=edain_id_clip_f${FOLD:-0}
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
# EDAIN: anchor=identity + outlier_clip=percentile
# (= AB configuration from the old lipo_nyul_v1 experiments, the best 0.8033)
# Now reimplemented as nnUNetTrainerEDAIN, so EVERYTHING below the EDAIN
# layer (augmentation, optimizer, loss, val protocol) is identical to vanilla
# nnU-Net. Any dice difference vs vanilla is caused by the EDAIN layer.
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

# Make our project importable so nnUNetv2_train can find nnUNetTrainerEDAIN.
PROJECT_ROOT="$HOME/MRI_EDAIN_2"   # adjust to actual cluster path
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# EDAIN knobs:
export EDAIN_ANCHOR_TYPE=identity
export EDAIN_OUTLIER_CLIP=percentile

FOLD=${FOLD:-0}
DATASET_ID=500
CONFIG=3d_fullres

echo "[EDAIN id+clip] fold=$FOLD | start $(date)"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
    -tr nnUNetTrainerEDAIN \
    --npz
echo "[EDAIN id+clip] fold=$FOLD | end $(date)"
