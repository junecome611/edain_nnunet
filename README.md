# edain_nnunet

**MRI-EDAIN v2: input-conditional differentiable monotonic spline normalization layer, integrated with the upstream nnU-Net v2 framework.**

This repository packages everything required to reproduce the EDAIN ablation experiments on the Lipo soft-tissue tumor MRI dataset (and any other MRI segmentation dataset using nnU-Net's standard pipeline).

---

## Repository structure — color-coded by origin

```
edain_nnunet/                             <repository root>
│
├── third_party_nnunet/                   🟦 UPSTREAM (MIC-DKFZ/nnUNet v2.7.0)
│   │                                        Vendored unmodified copy.
│   │                                        License: Apache 2.0 (see ./LICENSE inside).
│   ├── nnunetv2/...                      Original nnU-Net source.
│   ├── LICENSE                           Original Apache 2.0 license.
│   └── pyproject.toml                    Original (version = 2.7.0).
│
├── mri_edain_v2/                         🟩 OURS — EDAIN CORE LAYER
│   ├── modules/
│   │   ├── rq_spline.py                  ★ Rational-quadratic monotone spline (Durkan 2019, NeurIPS reference).
│   │   ├── hypernetwork.py               ★ 11→64→64→26 MLP with Fixup zero-init.
│   │   ├── percentile.py                 ★ 11-landmark γ computation.
│   │   ├── standardizer.py               ★ Per-coordinate γ standardization.
│   │   ├── nyul_init.py                  🔴 ★★ NYUL ANCHOR LOGIC (population_nyul + identity targets, L-BFGS fit).
│   │   ├── edain_layer.py                ★ MRIEDAINLayer top-level module.
│   │   └── foreground.py                 ★ Mask extractors (nonzero/Otsu/auto).
│   ├── losses/                           ★ Anchor + KL + combined losses.
│   ├── training/                         ★ Precompute + EMA + lambda scheduler.
│   ├── baselines/                        ★ P2 kill-switch (affine hypernet).
│   └── tests/                            ★ 26 unit tests.
│
├── edain_nnunet/                         🟨 OURS — nnU-Net INTEGRATION GLUE
│   ├── trainers/nnUNetTrainerEDAIN.py    ★ Subclass of nnUNetTrainer (~120 lines, 3 method overrides).
│   ├── network/edain_wrapper.py          ★ nn.Module wrapping EDAIN + nnU-Net backbone.
│   ├── precompute/nnunet_precompute.py   ★ Reads nnU-Net's preprocessed .npz, fits θ_0.
│   ├── scripts/                          ★ Slurm sbatch files (one per experiment).
│   └── docs/                             ★ Design + experiment matrix + diff vs upstream.
│
└── helpers/                              🟨 OURS — DATASET PLUMBING
    ├── prepare_nnunet_lipo.py            ★ Build raw symlinks for Dataset500_Lipo.
    ├── make_nnunet_splits.py             ★ Convert lipo_split.json → splits_final.json.
    └── lipo_split.json                   ★ Fixed 5-fold CV (114 cases, Lipo-073 excluded).
```

### Legend

- 🟦 **UPSTREAM** — not ours. Apache 2.0. Reviewing this is only necessary to confirm version + license.
- 🟩 **OURS — EDAIN core** — the differentiable normalization layer itself. Independent of nnU-Net.
- 🔴 **NYUL ANCHOR** — the Nyúl-inspired θ_0 fitting logic (inside `mri_edain_v2/modules/nyul_init.py`).
- 🟨 **OURS — integration / plumbing** — code that ties EDAIN into nnU-Net's training framework, plus dataset helpers.

---

## What to review

For a complete code review of "everything that isn't nnU-Net":

1. **`mri_edain_v2/`** (~3000 LOC, 350 KB) — the EDAIN layer + Nyúl anchor fitting + losses
2. **`edain_nnunet/`** (~600 LOC, 68 KB) — the integration code
3. **`helpers/`** (~100 LOC) — dataset preparation

`third_party_nnunet/` you only need to confirm: version = 2.7.0, license = Apache 2.0.

See `docs/CODE_INVENTORY.md` for a per-file breakdown.
See `docs/REVIEW_CHECKLIST.md` for a structured review pass.

---

## Installation (cluster)

```bash
# 1. Install nnU-Net 2.7.0 (the same version we vendor for reference)
pip install "nnunetv2==2.7.0" "blosc2<3"

# 2. Add this repo to PYTHONPATH so nnUNetv2_train can find nnUNetTrainerEDAIN
git clone https://github.com/junecome611/edain_nnunet.git
cd edain_nnunet
export PYTHONPATH=$PWD:$PYTHONPATH

# 3. Prepare data (Dataset500_Lipo)
cd helpers && python prepare_nnunet_lipo.py
nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity -c 3d_fullres
python make_nnunet_splits.py
```

## Run experiments

```bash
# vanilla nnU-Net (reference)
sbatch --export=FOLD=0 edain_nnunet/scripts/train_nnunet_vanilla.sh

# EDAIN with identity anchor + outlier clip (best config from prior runs, 0.8033)
sbatch --export=FOLD=0 edain_nnunet/scripts/train_edain_identity_clip.sh

# Plan A (popnyul + clip, expected worst)
sbatch --export=FOLD=0 edain_nnunet/scripts/train_edain_popnyul_clip.sh
```

See `edain_nnunet/docs/experiment_matrix.md` for the full 5-cell ablation design.

---

## Why this layout

The whole purpose is **causal attribution**: any Dice difference between vanilla nnU-Net and an EDAIN run must come from the EDAIN layer, because every other component (augmentation, optimizer, loss, schedule, validation protocol) is inherited from `nnUNetTrainer` unmodified.

See `edain_nnunet/docs/what_changed_vs_nnunet.md` for the exhaustive list of differences (spoiler: 3 method overrides totaling ~120 lines).

---

## License

- `third_party_nnunet/`: **Apache 2.0** (see `third_party_nnunet/LICENSE`, © MIC-DKFZ / Helmholtz Imaging)
- `mri_edain_v2/`, `edain_nnunet/`, `helpers/`: **MIT** (see `LICENSE` at repo root)
