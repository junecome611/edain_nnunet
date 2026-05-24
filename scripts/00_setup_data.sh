#!/bin/bash
#SBATCH --job-name=edain_setup
#SBATCH --partition=short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Only gpu005/gpu006 are known-good; other nodes have broken GPUs.
#SBATCH --nodelist=gpu[005-006]
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# One-time setup: prepare data + 2 preprocessing variants.
# Run this once before any training scripts.
#
# Outputs:
#   $nnUNet_raw/Dataset500_Lipo/                    (raw symlinks)
#   $nnUNet_preprocessed/Dataset500_Lipo/
#       ├── nnUNetPlans.json                        (z-scored — default)
#       ├── nnUNetPlans_3d_fullres/                 (z-scored .npz)
#       ├── nnUNetPlans_raw.json                    (no normalization)
#       └── nnUNetPlans_raw_3d_fullres/             (raw .npz)
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Python/3.11.5-GCCcore-13.2.0
source ~/nnunet_env/bin/activate

# Ensure nnunetv2 is installed (pin to 2.7.0 for reproducibility)
if ! python -c "import nnunetv2" 2>/dev/null; then
    echo "[setup] installing nnunetv2==2.7.0 + blosc2<3"
    pip install "blosc2<3" "nnunetv2==2.7.0"
fi
python -c "import nnunetv2, importlib.metadata as m; print('nnunetv2:', m.version('nnunetv2'))"

# nnU-Net data dirs (adjust if needed)
export nnUNet_raw=${nnUNet_raw:-$HOME/nnUNet_data/raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnUNet_data/results}
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

DATASET_ID=500
DATASET_NAME=Dataset500_Lipo
CONFIG=3d_fullres
REPO_ROOT="$SLURM_SUBMIT_DIR/.."  # this script is in scripts/
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- Step 1: raw symlinks ---
echo "[step 1] prepare raw data symlinks"
python "$REPO_ROOT/helpers/prepare_nnunet_lipo.py"

# --- Step 2: default (z-scored) preprocessing ---
FP_FLAG="$nnUNet_preprocessed/$DATASET_NAME/dataset_fingerprint.json"
if [ -f "$FP_FLAG" ]; then
    echo "[step 2] default plans/preprocessing already done, skip"
else
    echo "[step 2] nnUNetv2_plan_and_preprocess (z-score)"
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $CONFIG
fi

# --- Step 3: inject splits_final.json ---
echo "[step 3] writing splits_final.json from lipo_split.json"
python "$REPO_ROOT/helpers/make_nnunet_splits.py"

# --- Step 4: build raw plans (NoNormalization) ---
RAW_PLANS="$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans_raw.json"
if [ -f "$RAW_PLANS" ]; then
    echo "[step 4] raw plans already exist, skip"
else
    echo "[step 4] generating nnUNetPlans_raw"
    python -m edain_nnunet.plans.make_raw_plans -d $DATASET_ID \
        --src_plans_name nnUNetPlans \
        --dst_plans_name nnUNetPlans_raw \
        --configurations $CONFIG
fi

# --- Step 5: raw preprocessing ---
RAW_DIR="$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans_raw_$CONFIG"
if [ -d "$RAW_DIR" ] && [ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
    echo "[step 5] raw preprocessing already done, skip"
else
    echo "[step 5] preprocess with nnUNetPlans_raw"
    nnUNetv2_preprocess -d $DATASET_ID -c $CONFIG -plans_name nnUNetPlans_raw
fi

# --- Step 6: precompute EDAIN v1 stats for all 5 folds ---
STATS_DIR="$REPO_ROOT/edain_v1_stats"
mkdir -p "$STATS_DIR"
for f in 0 1 2 3 4; do
    OUT_JSON="$STATS_DIR/edain_v1_stats_fold${f}.json"
    if [ -f "$OUT_JSON" ]; then
        echo "[step 6] fold $f stats already exist, skip"
        continue
    fi
    echo "[step 6] precomputing EDAIN v1 stats for fold $f"
    python -m mri_edain_v1.precompute.precompute_v1_stats \
        --preprocessed_dir "$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans_raw_$CONFIG" \
        --splits_json     "$nnUNet_preprocessed/$DATASET_NAME/splits_final.json" \
        --output_json     "$OUT_JSON" \
        --fold $f
done

echo "[setup] done at $(date)"
echo "Stats live in: $STATS_DIR"
