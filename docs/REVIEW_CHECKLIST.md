# Review Checklist (给用户用的)

> 用这个 checklist 系统地过一遍**所有非 nnU-Net 代码**。每项打勾后写 OK/NOTES。

---

## 0. 跳过的部分 (UPSTREAM)

- [ ] `third_party_nnunet/pyproject.toml` 显示 version = 2.7.0
- [ ] `third_party_nnunet/LICENSE` 显示 Apache 2.0
- [ ] **以下无需 review**：`third_party_nnunet/nnunetv2/**`（这就是上游 nnU-Net）

---

## 1. 🟩 EDAIN core 数学正确性

### `mri_edain_v2/modules/rq_spline.py`

- [ ] `rq_spline_parameterize`：
  - 输入 `(B, 3K-1)` 维 logits → 输出 SplineParams
  - widths/heights 通过 softmax 保证 sum = 2*B_supp
  - 内部 derivatives 通过 softplus + min_derivative 保证 > 0
  - 边界 derivatives = alpha_tail (= 0.5)
  - **检查点**：min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3 防止数值病态
- [ ] `rq_spline_apply`：
  - 输入位于 [-B, +B] 内：用 Durkan 2019 Eq.4 计算
  - 输入超出 [-B, +B]：线性外推（slope = alpha_tail）
  - 严格单调（保序）

### `mri_edain_v2/modules/hypernetwork.py`

- [ ] 网络结构：`11 → 64 (GELU) → 64 (GELU) → 26`
- [ ] **关键**：最后一层 `W_3` 必须**零初始化** + `b_3` 零初始化（Fixup 原则）
  - 这样训练开始时 Δθ = 0，spline = θ_0 anchor
- [ ] 中间层用 Kaiming init

### `mri_edain_v2/modules/percentile.py`

- [ ] `PERCENTILES = (0.01, 0.10, ..., 0.99)` — 11 个 Shah 2011 landmark
- [ ] `percentile_summary` 在 foreground 上用 `torch.quantile`
- [ ] **detached** —— 梯度不流过 percentile（避免 double-moving-target）
- [ ] 大于 2^24 voxel 时做 randint 子采样（torch.quantile 上限）

### `mri_edain_v2/modules/standardizer.py`

- [ ] `CoordinateStandardizer` 用 fit() 一次性算 μ/σ，存成 frozen buffer
- [ ] forward: `(γ - μ) / (σ + eps)`，输出和输入同 dtype
- [ ] **buffer，不是 parameter**——不参与优化

---

## 2. 🔴 NYUL 锚点逻辑 (重点 review)

### `mri_edain_v2/modules/nyul_init.py`

- [ ] **anchor_type 两种**：
  - `population_nyul`: 把训练集 11 landmark L_p → Φ⁻¹(p) 标准正态分位
  - `identity`: target = L 本身（保证 spline 起点 ≈ y=x）
- [ ] **L-BFGS 拟合**：
  - 注意 `fit_mask = (grid >= L[0]) & (grid <= L[-1])` — **只在 landmark 范围内拟合**
  - 这是修复 bug "loss=1075"的关键
- [ ] piecewise_linear_interp 用 `searchsorted` + `idx.clamp(0, L-2)`，linear extrapolation outside
- [ ] `compute_non_affineness`：在 200 点网格上算 SS_res / SS_tot 的开根

---

## 3. 🟩 EDAIN 顶层组装

### `mri_edain_v2/modules/edain_layer.py`

- [ ] **forward 流程**：
  1. 算 gamma_raw（detached）
  2. standardize → gamma_std
  3. hypernet(gamma_std) → Δθ
  4. θ = θ_0 + Δθ
  5. 参数化 → SplineParams
  6. 在 foreground voxel 上 apply spline
  7. **background voxel 原样保留**（`torch.where(mask, X_tilde_fg, X_b)`）
- [ ] 支持 (D,H,W) / (B,D,H,W) / (B,1,D,H,W) 三种输入
- [ ] **重要**：如果 caller 传 `gamma_raw` 就用 caller 的，**不重算**（这是修复 bug A1 的入口）

---

## 4. 🟩 Loss 函数

### `mri_edain_v2/losses/anchor.py`

- [ ] Function-space anchor: `||f_θ(t_l) - f_θ_0(t_l)||²` mean over 50 grid points
- [ ] **不是** parameter-space (`||θ - θ_0||²`)—这是准则 P3

### `mri_edain_v2/losses/kl.py`

- [ ] Soft Gaussian histogram，可微
- [ ] Subsample 50000 voxels per batch（避免 OOM）
- [ ] Target distribution = N(0,1) discrete pmf over bins

### `mri_edain_v2/losses/combined.py`

- [ ] `L = L_seg + λ_anc · L_anc + λ_kl · L_kl`
- [ ] 输出 dataclass 含 detached components 用于 logging

---

## 5. 🟩 Training utilities

### `mri_edain_v2/training/precompute.py`

- [ ] Per-case 应用上游预处理 → 算 γ
- [ ] Fit standardizer 和 θ_0
- [ ] 保存 PrecomputeArtifacts (含 raw_gammas, population_landmarks, etc.)
- [ ] 支持 outlier_clip='percentile' / 'none'
- [ ] 支持 anchor_type='population_nyul' / 'identity'

### `mri_edain_v2/training/scheduler.py`

- [ ] 三相调度（phase_0_end, phase_1_end, total_steps）
- [ ] phase 0：λ_anc=init, λ_kl=0
- [ ] phase 1：同 phase 0
- [ ] phase 2：λ_anc cosine decay, λ_kl linear ramp
- [ ] **允许 phase_*_end > total_steps**（用来表达"forever frozen"）

### `mri_edain_v2/training/ema.py`

- [ ] EMA decay = 0.99
- [ ] swap_in/swap_out context manager 用于 validation

---

## 6. 🟨 nnU-Net 集成

### `edain_nnunet/network/edain_wrapper.py`

- [ ] `EDAINWrapper(edain, backbone, case_gamma_table)`
- [ ] `forward(x)`：
  - 如果 `case_gamma_table` + `_current_case_ids` 都有 → 用表 lookup（**修 bug A1**）
  - 否则在 patch 上算（fallback）
- [ ] `set_current_batch(case_ids)` 由 trainer 在 forward 之前调

### `edain_nnunet/precompute/nnunet_precompute.py`

- [ ] 直接读 `$nnUNet_preprocessed/{ds}/{plans}_{config}/{case}.npz` 的 `data[0]`
- [ ] **复用 nnU-Net 真正的预处理输出**，避免重新实现 MONAI pipeline
- [ ] 返回 dict 含 case_gammas（**这是 case_id → 整卷 γ 查表的源**）

### `edain_nnunet/trainers/nnUNetTrainerEDAIN.py`

- [ ] `initialize()`：先 `super().initialize()`，然后 precompute + wrap network + **重建 optimizer**
- [ ] `train_step` / `validation_step`：把 `batch["keys"]` 传给 wrapper 后调用 `super().train_step()`
- [ ] **环境变量**：`EDAIN_ANCHOR_TYPE`, `EDAIN_OUTLIER_CLIP`（避免改 CLI）

---

## 7. 🟨 Scripts (Slurm)

每个 .sh 检查：

- [ ] `module load Python/3.11.5-GCCcore-13.2.0` + `source ~/nnunet_env/bin/activate`
- [ ] 设置 `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`
- [ ] `nnUNet_n_proc_DA=0` 和 `nnUNet_compile=F` 防止 fork-after-cuda 崩
- [ ] `export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH` 让 nnUNetv2_train 找到我们的 trainer
- [ ] `EDAIN_ANCHOR_TYPE` 和 `EDAIN_OUTLIER_CLIP` 设置正确
- [ ] 调用 `nnUNetv2_train 500 3d_fullres $FOLD -tr nnUNetTrainerEDAIN --npz`

---

## 8. 🟨 Helpers

### `helpers/prepare_nnunet_lipo.py`

- [ ] 排除 Lipo-073（multi-tumor）
- [ ] 建符号链接到 `$nnUNet_raw/Dataset500_Lipo/imagesTr/` 和 `labelsTr/`
- [ ] 文件名格式：`Lipo-XXX_MR_1_0000.nii.gz`（_0000 表示 channel 0）

### `helpers/make_nnunet_splits.py`

- [ ] 读 `lipo_split.json`
- [ ] 写到 `$nnUNet_preprocessed/Dataset500_Lipo/splits_final.json`
- [ ] 5 个 fold，每个 fold 含 `{"train": [...], "val": [...]}`

---

## 9. 实验设计

- [ ] `edain_nnunet/docs/experiment_matrix.md` 列了 5 个对照
- [ ] Sanity check 期望：Exp #1 (vanilla) 应该 ≈ `results/lipo_nnunet/` 的 0.8254
- [ ] AB 配置 (Exp #3) 应该复现 ≥ 0.80（之前 0.8033）

---

## 总结

- 全部 OK：可以开始跑实验
- 有问题：在每项下面写 NOTES 然后告诉我

预计 review 时长：~1-2 小时（针对 ~3400 LOC 我们自己写的代码）。
