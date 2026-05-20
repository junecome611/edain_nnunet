# edain_nnunet

Learnable MRI intensity normalization (EDAIN v1 + Nyúl-style spline) integrated with the upstream nnU-Net v2 framework.

This repo packages **two independent learnable-normalization designs** and lets you compare them head-to-head against the vanilla nnU-Net baseline:

| Design  | Where it lives | What it is |
|---|---|---|
| 🔵 **EDAIN v1** | `mri_edain_v1/` | The 4-sublayer EDAIN from Sanna Passino et al. 2024 — `h1` (tanh outlier) + `h2` (shift) + `h3` (scale) + `h4` (Yeo-Johnson power transform). |
| 🟣 **Nyul** | `mri_edain_v2/` | Our newer design — input-conditional rational-quadratic monotone spline whose anchor is fit from training-set Nyúl landmarks. |

Both run on the **same data, same augmentation, same optimizer, same backbone** as upstream nnU-Net — the only difference is the normalization layer that sits in front of the network. This makes any Dice difference causally attributable to that layer.

---

## Repository structure (color-coded)

```
edain_nnunet/                                       <repo root>
├── third_party_nnunet/      🟦 UPSTREAM            nnU-Net v2.7.0 (unmodified, Apache 2.0)
│
├── mri_edain_v1/            🔵 OURS — EDAIN (paper)
│   ├── modules/edain_v1_layer.py        4 sublayers, local-aware, optional h4
│   ├── modules/yeo_johnson.py           h4 power transform (Yeo-Johnson)
│   ├── modules/percentile_stats.py      per-case fg_mean/fg_std/fg_p2/fg_p98
│   └── precompute/precompute_v1_stats.py
│
├── mri_edain_v2/            🟣 OURS — Nyul (ours)
│   ├── modules/rq_spline.py             monotone RQ-spline (Durkan 2019)
│   ├── modules/hypernetwork.py          11 -> 64 -> 64 -> 26 MLP (Fixup zero-init)
│   ├── modules/nyul_init.py             theta_0 anchor fitting (population_nyul / identity)
│   ├── modules/edain_layer.py           MRIEDAINLayer top-level module
│   └── ... (losses, training utilities, tests)
│
├── edain_nnunet/            🟨 INTEGRATION GLUE
│   ├── trainers/
│   │   ├── nnUNetTrainerEDAINv1.py          uses EDAIN v1 (h1+h2+h3)
│   │   ├── nnUNetTrainerEDAINv1Power.py     uses EDAIN v1 + h4
│   │   └── nnUNetTrainerNyul.py             uses Nyul spline
│   ├── network/             nn.Module wrappers
│   ├── precompute/          Nyul artifact builder
│   └── plans/make_raw_plans.py             generates nnUNetPlans_raw (NoNormalization)
│
├── helpers/                 🟨 DATASET HELPERS
│   ├── prepare_nnunet_lipo.py             build Dataset500_Lipo symlinks
│   ├── make_nnunet_splits.py              write splits_final.json from lipo_split.json
│   └── lipo_split.json                    fixed 5-fold CV
│
├── scripts/                 🟨 SLURM JOBS (one per experiment)
│   ├── 00_setup_data.sh                  ONE-TIME setup (data + 2 preprocessings + precompute)
│   ├── 01_baseline_nnunet.sh             EXP 0: vanilla nnU-Net (z-score)
│   ├── 02_edain_v1.sh                    EXP 1: EDAIN v1 (h1+h2+h3, no power)
│   ├── 03_edain_v1_power.sh              EXP 2: EDAIN v1 + power (h1+h2+h3+h4)
│   ├── 04_nyul_identity.sh               EXP 3: Nyul spline, identity anchor
│   └── 05_nyul_popnyul.sh                EXP 4: Nyul spline, population_nyul anchor
│
└── docs/                                  CODE_INVENTORY, REVIEW_CHECKLIST, design.md
```

---

## Quick start (on the cluster)

```bash
# 1. Clone
cd $HOME
git clone https://github.com/junecome611/edain_nnunet.git
cd edain_nnunet

# 2. (Optional) configure where nnU-Net stores data
export nnUNet_raw=$HOME/nnUNet_data/raw
export nnUNet_preprocessed=$HOME/nnUNet_data/preprocessed
export nnUNet_results=$HOME/nnUNet_data/results

# 3. ONE-TIME setup: prepare data + both preprocessing variants + precompute EDAIN v1 stats
sbatch scripts/00_setup_data.sh

# Wait until 00 finishes, then submit experiments (1 fold each):
sbatch --export=ALL,FOLD=0 scripts/01_baseline_nnunet.sh        # vanilla nnU-Net
sbatch --export=ALL,FOLD=0 scripts/02_edain_v1.sh               # EDAIN v1 (no power)
sbatch --export=ALL,FOLD=0 scripts/03_edain_v1_power.sh         # EDAIN v1 + power
sbatch --export=ALL,FOLD=0 scripts/04_nyul_identity.sh          # Nyul identity
sbatch --export=ALL,FOLD=0 scripts/05_nyul_popnyul.sh           # Nyul popnyul

# For 5-fold: submit each with FOLD=0,1,2,3,4
for f in 0 1 2 3 4; do
    sbatch --export=ALL,FOLD=$f scripts/02_edain_v1.sh
done
```

---

## Experiment matrix

| # | Script | Trainer | Plans | What it tests |
|---|---|---|---|---|
| 0 | `01_baseline_nnunet.sh` | `nnUNetTrainer` (upstream) | `nnUNetPlans` (z-score) | reference baseline |
| 1 | `02_edain_v1.sh` | `nnUNetTrainerEDAINv1` | `nnUNetPlans_raw` | does EDAIN v1 (no power) beat fixed z-score? |
| 2 | `03_edain_v1_power.sh` | `nnUNetTrainerEDAINv1Power` | `nnUNetPlans_raw` | does the YJ power transform (h4) help? |
| 3 | `04_nyul_identity.sh` | `nnUNetTrainerNyul` | `nnUNetPlans` (z-score) | Nyul framework with neutral identity start |
| 4 | `05_nyul_popnyul.sh` | `nnUNetTrainerNyul` | `nnUNetPlans` (z-score) | Nyul framework with classical Nyul anchor |

**EXP 1 vs EXP 2** isolates the contribution of the power transform.
**EXP 3 vs EXP 4** isolates the contribution of the Nyúl-style anchor.
**EXP 0** is the absolute reference (prior 5-fold mean ≈ 0.779 on Lipo).

---

## Initialization values (calibrated on Lipo training set)

EDAIN v1 (see `notebooks/edain_init_analysis/` in the parent project for derivation):
```
init_alpha  = 0.5    h1 outlier-mit blend ratio (50/50 winsorize:identity at start)
init_beta   = 1.5    tanh transition on [0,1] rescaled input
init_m      = 0.0    no extra shift (rescaled input is already centered ~0.5)
init_s      = 1.0    no extra scale (rescaled input is already in [0,1])
init_lambda = 1.0    YJ(x; 1) = x  (h4 is identity at start, only learned deviates)
beta_min    = 0.1    softplus lower bound
```

These come with a fixed per-image rescale `(x − fg_p2) / (fg_p98 − fg_p2)` to `~[0,1]` BEFORE `h1` so β is in its natural scale. See `mri_edain_v1/modules/edain_v1_layer.py` for the full forward pass.

---

## Per-sublayer learning rate (paper recommendation)

Default LR modifiers (overridable via env vars):
```
EDAIN_V1_LR_ALPHA = 10       outlier sublayer
EDAIN_V1_LR_BETA  = 10       outlier sublayer
EDAIN_V1_LR_SHIFT = 1        shift sublayer
EDAIN_V1_LR_SCALE = 1        scale sublayer
EDAIN_V1_LR_POWER = 10       Yeo-Johnson sublayer
```
Backbone uses the nnU-Net default `initial_lr = 1e-2` SGD. EDAIN params use `1e-2 × multiplier`.

---

## License

- `third_party_nnunet/`: Apache 2.0 (© MIC-DKFZ / Helmholtz Imaging)
- Everything else: MIT (see `LICENSE`)

See `docs/CODE_INVENTORY.md` for a per-file ownership breakdown and `docs/REVIEW_CHECKLIST.md` for a structured review pass.
