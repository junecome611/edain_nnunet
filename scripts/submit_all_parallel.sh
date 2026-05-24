#!/bin/bash
# ============================================================
# Submit the 5 experiments as INDEPENDENT parallel SLURM jobs.
#
# Each sbatch reserves --gres=gpu:1 and --nodelist=gpu[005-006], so they
# share the 2 known-good GPU nodes. SLURM queues them as resources free up.
# All 5 trainers write to DIFFERENT nnU-Net output dirs (different
# __TrainerClass__ component of the path), so they cannot collide.
#
# Each per-experiment script auto-resumes from checkpoint_latest.pth if
# present, so cancel+resubmit is safe.
#
# USAGE
#     bash scripts/submit_all_parallel.sh           # all 5 experiments, fold 0
#     bash scripts/submit_all_parallel.sh 1          # all 5 experiments, fold 1
#     bash scripts/submit_all_parallel.sh 0 1 2      # 3 folds in parallel * 5 exps
#
# PRECONDITION
#     scripts/run_all_fold0.sh has been run AT LEAST ONCE successfully past
#     the setup phase (data prepared, raw plans + raw preprocessing exist,
#     edain_v1_stats_fold0.json exists). The training step doesn't matter —
#     we just need the setup state. If unsure, run:
#         sbatch scripts/run_all_fold0.sh
#     and wait until the setup section reports done in the log.
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root

FOLDS="${@:-0}"           # default: just fold 0
if [ -z "$FOLDS" ]; then
    FOLDS="0"
fi

# Sanity: warn (don't fail) if expected setup artifacts are missing.
DEFAULT_DIR="${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}/Dataset500_Lipo/nnUNetPlans_3d_fullres"
RAW_DIR="${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}/Dataset500_Lipo/nnUNetPlans_raw_3d_fullres"
echo "[parallel] Expecting setup artifacts at:"
echo "             $DEFAULT_DIR (default z-scored)"
echo "             $RAW_DIR (raw)"
for d in "$DEFAULT_DIR" "$RAW_DIR"; do
    if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "[parallel] WARNING: $d missing or empty. Run scripts/run_all_fold0.sh first."
        echo "[parallel] Press Ctrl-C within 5s to abort."
        sleep 5
        break
    fi
done

submit() {
    local fold="$1"
    local script="$2"
    echo "[parallel] sbatch FOLD=$fold $script"
    sbatch --export=ALL,FOLD=$fold "scripts/$script"
}

for f in $FOLDS; do
    submit "$f" "01_baseline_nnunet.sh"
    submit "$f" "02_edain_v1.sh"
    submit "$f" "03_edain_v1_power.sh"
    submit "$f" "04_nyul_identity.sh"
    submit "$f" "05_nyul_popnyul.sh"
done

echo
echo "[parallel] queue status:"
squeue -u "$USER" --format="%.10i %.20j %.8T %.10M %.4D %R" 2>/dev/null || true
