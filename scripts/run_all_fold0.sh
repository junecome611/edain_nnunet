#!/bin/bash
#SBATCH --job-name=edain_all_f0
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
# Master driver: setup + 5 experiments on fold 0, one job.
#
# RUNTIME EXPECTATIONS
# --------------------
#   Setup:          ~1-2h   (one-time, skipped on resubmit)
#   Each experiment ~24-45h (depending on A40 vs 2080Ti)
#   Total:          ~120-225h sequentially
#
# Partition wall-time is 4 days 12 hours = 108 hours. That fits the setup
# plus 2-3 experiments. The script is FULLY IDEMPOTENT and RESUMABLE:
#   - Setup steps detect existing artifacts and skip.
#   - Each experiment writes its checkpoint to a unique nnU-Net output dir
#     (driven by the trainer class name + plans + config).
#   - On resubmit we detect:
#       * checkpoint_final.pth exists  -> skip experiment
#       * checkpoint_latest.pth exists -> resume with `--c`
#       * nothing                       -> start fresh
#
# WHAT YOU SHOULD DO
# ------------------
#   Submit:
#       sbatch scripts/run_all_fold0.sh
#   When the job times out (likely 2-3 times), just resubmit the same line.
#   Each resubmit picks up where it left off; experiments already done are
#   skipped instantly.
#
# REQUIRED ENV VARS (set before sbatch, or rely on defaults below)
# ----------------------------------------------------------------
#   nnUNet_raw, nnUNet_preprocessed, nnUNet_results
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ----- environment -----
module load Python/3.11.5-GCCcore-13.2.0
source ~/nnunet_env/bin/activate

export nnUNet_raw=${nnUNet_raw:-$HOME/nnUNet_data/raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnUNet_data/preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnUNet_data/results}
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Stable nnU-Net runtime knobs: same ones as the per-experiment scripts.
export nnUNet_n_proc_DA=0
export nnUNet_compile=F

# Make our repo modules importable so `nnUNetv2_train -tr nnUNetTrainerEDAINv1`
# can find the trainer classes.
REPO_ROOT="$SLURM_SUBMIT_DIR/.."
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FOLD=0
DATASET_ID=500
DATASET_NAME=Dataset500_Lipo
CONFIG=3d_fullres

# Where nnU-Net writes its per-experiment output folders.
RESULTS_BASE="$nnUNet_results/$DATASET_NAME"

# ----- helper: skip-or-resume one experiment -----
# Args: <step_name> <trainer_class> <plans_identifier> [extra env exports as `KEY=VAL`...]
run_experiment () {
    local step_name="$1"; shift
    local trainer="$1";   shift
    local plans="$1";     shift
    # Remaining args are KEY=VAL exports we should apply for this experiment only.
    # We run them in a subshell to avoid leaking into later experiments.

    local out_dir="$RESULTS_BASE/${trainer}__${plans}__${CONFIG}/fold_${FOLD}"
    local final="$out_dir/checkpoint_final.pth"
    local latest="$out_dir/checkpoint_latest.pth"

    if [ -f "$final" ]; then
        echo "[$step_name] DONE ($final exists), skipping."
        return 0
    fi

    echo "==========================================================="
    echo "[$step_name] start at $(date) | trainer=$trainer plans=$plans"
    echo "==========================================================="

    (
        # Apply experiment-local env vars (after the `--`-style KEY=VAL args).
        for kv in "$@"; do
            export "$kv"
            echo "[$step_name] env: $kv"
        done

        if [ -f "$latest" ]; then
            echo "[$step_name] resuming from $latest"
            nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
                -tr "$trainer" -p "$plans" --npz --c
        else
            echo "[$step_name] starting fresh"
            nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
                -tr "$trainer" -p "$plans" --npz
        fi
    )

    if [ -f "$final" ]; then
        echo "[$step_name] FINISHED at $(date)"
    else
        echo "[$step_name] PAUSED at $(date)  (likely time-limit hit; resubmit to continue)"
        exit 0   # exit cleanly so the next resubmit picks up here
    fi
}

# ============================================================
# STEP 0  --  one-time data preparation
# ============================================================
echo "==========================================================="
echo "STEP 0: data setup at $(date)"
echo "==========================================================="

# Install nnunetv2 if missing
if ! python -c "import nnunetv2" 2>/dev/null; then
    echo "[setup] installing nnunetv2==2.7.0 + blosc2<3"
    pip install "blosc2<3" "nnunetv2==2.7.0"
fi
python -c "import nnunetv2, importlib.metadata as m; print('nnunetv2:', m.version('nnunetv2'))"

# Register our trainers into nnU-Net's search path (idempotent).
# Without this nnUNetv2_train errors with "Could not find requested nnunet trainer".
echo "[setup] registering EDAIN/Nyul trainers into nnU-Net's search path"
python -m tools.register_trainers

# Symlinks
if [ ! -d "$nnUNet_raw/$DATASET_NAME/imagesTr" ] \
   || [ -z "$(ls -A "$nnUNet_raw/$DATASET_NAME/imagesTr" 2>/dev/null)" ]; then
    echo "[setup] preparing raw symlinks"
    python "$REPO_ROOT/helpers/prepare_nnunet_lipo.py"
else
    echo "[setup] raw symlinks already exist, skip"
fi

# Default (z-score) preprocessing
FP_FLAG="$nnUNet_preprocessed/$DATASET_NAME/dataset_fingerprint.json"
if [ -f "$FP_FLAG" ]; then
    echo "[setup] default preprocessing already done, skip"
else
    echo "[setup] running nnUNetv2_plan_and_preprocess (z-score)"
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $CONFIG
fi

# splits_final.json
SPLITS="$nnUNet_preprocessed/$DATASET_NAME/splits_final.json"
if [ ! -f "$SPLITS" ]; then
    echo "[setup] writing splits_final.json"
    python "$REPO_ROOT/helpers/make_nnunet_splits.py"
else
    echo "[setup] splits_final.json already exists, skip"
fi

# Raw (NoNormalization) plans
RAW_PLANS="$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans_raw.json"
if [ ! -f "$RAW_PLANS" ]; then
    echo "[setup] generating nnUNetPlans_raw"
    python -m edain_nnunet.plans.make_raw_plans -d $DATASET_ID \
        --src_plans_name nnUNetPlans \
        --dst_plans_name nnUNetPlans_raw \
        --configurations $CONFIG
else
    echo "[setup] nnUNetPlans_raw already exists, skip"
fi

# Raw preprocessing
RAW_DIR="$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans_raw_$CONFIG"
if [ -d "$RAW_DIR" ] && [ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
    echo "[setup] raw preprocessing already done, skip"
else
    echo "[setup] running raw preprocessing"
    nnUNetv2_preprocess -d $DATASET_ID -c $CONFIG -plans_name nnUNetPlans_raw
fi

# EDAIN v1 per-case stats (fold 0 only — we run all 5 experiments on fold 0)
STATS_JSON="$REPO_ROOT/edain_v1_stats/edain_v1_stats_fold${FOLD}.json"
mkdir -p "$REPO_ROOT/edain_v1_stats"
if [ ! -f "$STATS_JSON" ]; then
    echo "[setup] precomputing EDAIN v1 stats for fold $FOLD"
    python -m mri_edain_v1.precompute.precompute_v1_stats \
        --preprocessed_dir "$RAW_DIR" \
        --splits_json     "$SPLITS" \
        --output_json     "$STATS_JSON" \
        --fold $FOLD
else
    echo "[setup] EDAIN v1 stats for fold $FOLD already exist, skip"
fi

echo "[setup] DONE at $(date)"

# ============================================================
# STEP 1  --  EXP 0: vanilla nnU-Net (z-score baseline)
# ============================================================
run_experiment "EXP 0 (vanilla nnU-Net)" \
    "nnUNetTrainer" "nnUNetPlans"

# ============================================================
# STEP 2  --  EXP 1: EDAIN v1 (h1+h2+h3, no power transform)
# ============================================================
run_experiment "EXP 1 (EDAIN v1)" \
    "nnUNetTrainerEDAINv1" "nnUNetPlans_raw" \
    "EDAIN_V1_STATS_JSON=$STATS_JSON" \
    "EDAIN_V1_USE_POWER=0" \
    "EDAIN_V1_RESCALE_P2P98=1" \
    "EDAIN_V1_LR_ALPHA=10" "EDAIN_V1_LR_BETA=10" \
    "EDAIN_V1_LR_SHIFT=1" "EDAIN_V1_LR_SCALE=1"

# ============================================================
# STEP 3  --  EXP 2: EDAIN v1 + Power Transform (h1+h2+h3+h4)
# ============================================================
run_experiment "EXP 2 (EDAIN v1 + power)" \
    "nnUNetTrainerEDAINv1Power" "nnUNetPlans_raw" \
    "EDAIN_V1_STATS_JSON=$STATS_JSON" \
    "EDAIN_V1_USE_POWER=1" \
    "EDAIN_V1_RESCALE_P2P98=1" \
    "EDAIN_V1_LR_ALPHA=10" "EDAIN_V1_LR_BETA=10" \
    "EDAIN_V1_LR_SHIFT=1" "EDAIN_V1_LR_SCALE=1" "EDAIN_V1_LR_POWER=10"

# ============================================================
# STEP 4  --  EXP 3: Nyul + identity anchor + clip
# ============================================================
run_experiment "EXP 3 (Nyul identity)" \
    "nnUNetTrainerNyulIdentity" "nnUNetPlans"

# ============================================================
# STEP 5  --  EXP 4: Nyul + population_nyul anchor + no clip
# ============================================================
run_experiment "EXP 4 (Nyul popnyul)" \
    "nnUNetTrainerNyulPopnyul" "nnUNetPlans"

echo "==========================================================="
echo "ALL 5 EXPERIMENTS FINISHED for fold $FOLD at $(date)"
echo "==========================================================="
echo
echo "Final checkpoints:"
for trainer_plans in \
    "nnUNetTrainer__nnUNetPlans" \
    "nnUNetTrainerEDAINv1__nnUNetPlans_raw" \
    "nnUNetTrainerEDAINv1Power__nnUNetPlans_raw" \
    "nnUNetTrainerNyulIdentity__nnUNetPlans" \
    "nnUNetTrainerNyulPopnyul__nnUNetPlans"; do
    f="$RESULTS_BASE/${trainer_plans}__${CONFIG}/fold_${FOLD}/checkpoint_final.pth"
    [ -f "$f" ] && echo "  OK   $f" || echo "  MISS $f"
done
