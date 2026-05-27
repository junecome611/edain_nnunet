#!/bin/bash
#SBATCH --job-name=precompute_edain_stats
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Re-run the EDAIN v1 per-case stats precompute for one or more folds.
#
# USAGE
#     sbatch scripts/recompute_edain_stats.sh           # default: fold 0
#     sbatch scripts/recompute_edain_stats.sh 0 1 2 3 4 # all 5 folds
#
# Output: $REPO_ROOT/edain_v1_stats/edain_v1_stats_fold<F>.json
# Each JSON contains per-case (fg_mean, fg_std, fg_p2, fg_p98) for BOTH
# train and val cases of that fold — needed for the train-eval bug fix.
#
# This is CPU-only (reads preprocessed .b2nd, computes percentiles).
# No GPU node required.
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

REPO_ROOT="$SLURM_SUBMIT_DIR/.."
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FOLDS="${@:-0}"
if [ -z "$FOLDS" ]; then
    FOLDS="0"
fi

PREPROC_DIR="$nnUNet_preprocessed/Dataset500_Lipo/nnUNetPlans_raw_3d_fullres"
SPLITS_JSON="$nnUNet_preprocessed/Dataset500_Lipo/splits_final.json"
OUT_DIR="$REPO_ROOT/edain_v1_stats"
mkdir -p "$OUT_DIR"

for F in $FOLDS; do
    echo "============================================================"
    echo "[precompute] fold $F  ($(date))"
    echo "============================================================"
    OUT_JSON="$OUT_DIR/edain_v1_stats_fold${F}.json"
    # Always remove the old file first — partial files from a crashed run
    # would otherwise be picked up by the trainer.
    rm -f "$OUT_JSON"
    python -m mri_edain_v1.precompute.precompute_v1_stats \
        --preprocessed_dir "$PREPROC_DIR" \
        --splits_json     "$SPLITS_JSON" \
        --fold            "$F" \
        --output_json     "$OUT_JSON"
    echo "[precompute] fold $F done -> $OUT_JSON"
done

echo "[precompute] all done $(date)"
