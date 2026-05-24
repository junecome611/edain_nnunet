#!/bin/bash
# ============================================================
# Nuke ALL state from prior (possibly broken) runs so that the
# next sbatch run_all_fold0.sh starts from a clean slate.
#
# Safe: only deletes things this repo's pipeline generates.
# Touches:
#   - nnUNet plans and preprocessed dirs for Dataset500_Lipo
#   - edain_v1_stats/
#   - nnUNet_results/Dataset500_Lipo/  (all 5 trainer dirs)
#
# DOES NOT touch:
#   - dataset/lipo/        (raw data)
#   - nnUNet_raw/          (symlinks)
#   - nnUNet_preprocessed/Dataset500_Lipo/splits_final.json
#   - nnUNet_preprocessed/Dataset500_Lipo/dataset.json
#   - the repo itself
#
# USAGE:
#   bash scripts/reset_state.sh        (interactive prompt)
#   bash scripts/reset_state.sh -y     (skip prompt)
# ============================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnUNet_data/results}

DATASET_NAME=Dataset500_Lipo
PRE_DIR="$nnUNet_preprocessed/$DATASET_NAME"
RES_DIR="$nnUNet_results/$DATASET_NAME"
STATS_DIR="$REPO_ROOT/edain_v1_stats"

echo "Will delete the following (if they exist):"
echo "  $PRE_DIR/nnUNetPlans*.json          (plans files)"
echo "  $PRE_DIR/nnUNetPlans*_3d_fullres    (preprocessed .npz dirs)"
echo "  $PRE_DIR/dataset_fingerprint.json   (forces plan_and_preprocess)"
echo "  $STATS_DIR/                          (EDAIN v1 per-case stats)"
echo "  $RES_DIR/nnUNetTrainer*              (all 5 trainer output dirs)"
echo
echo "PRESERVED:"
echo "  $PRE_DIR/splits_final.json"
echo "  $PRE_DIR/dataset.json"
echo "  $PRE_DIR/gt_segmentations/"
echo "  $nnUNet_raw/   (raw symlinks)"
echo

if [ "${1:-}" != "-y" ]; then
    read -p "Proceed? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "aborted."; exit 0 ;;
    esac
fi

echo
echo "[reset] removing plans + preprocessed dirs"
rm -f  "$PRE_DIR"/nnUNetPlans*.json     2>/dev/null || true
rm -rf "$PRE_DIR"/nnUNetPlans*_3d_fullres 2>/dev/null || true
rm -f  "$PRE_DIR"/dataset_fingerprint.json 2>/dev/null || true

echo "[reset] removing EDAIN v1 stats"
rm -rf "$STATS_DIR" 2>/dev/null || true

echo "[reset] removing all trainer output dirs (5 experiments)"
rm -rf "$RES_DIR"/nnUNetTrainer* 2>/dev/null || true

echo
echo "[reset] DONE. Next step:"
echo "    sbatch scripts/run_all_fold0.sh"
