# 代码清单 (Code Inventory)

> 这个文档列出仓库内**每一个文件**的归属、来源、行数和作用。
> 用 🟦 / 🟩 / 🔴 / 🟨 标识归属。

## 归属图例

| 标识 | 含义 | License |
|---|---|---|
| 🟦 UPSTREAM | nnU-Net 2.7.0 原版（未修改）| Apache 2.0 |
| 🟩 OURS — EDAIN core | 我们的 EDAIN 层 | MIT |
| 🔴 NYUL ANCHOR | 我们的 Nyúl 锚点拟合逻辑 | MIT |
| 🟨 OURS — integration | 我们的 nnU-Net 集成胶水代码 | MIT |

---

## 1. 🟦 UPSTREAM (third_party_nnunet/)

整个 `third_party_nnunet/` 文件夹是 [MIC-DKFZ/nnUNet v2.7.0](https://github.com/MIC-DKFZ/nnUNet) 的未修改副本。

| 路径 | 作用 |
|---|---|
| `third_party_nnunet/nnunetv2/` | nnU-Net Python 包 |
| `third_party_nnunet/LICENSE` | Apache 2.0 原始许可证 |
| `third_party_nnunet/pyproject.toml` | 版本声明（version = 2.7.0）|
| `third_party_nnunet/documentation/` | nnU-Net 官方文档 |

**修改总数：0 行**。我们用 pip 装 `nnunetv2==2.7.0` 在 cluster 上跑，本地仓库的副本是给 reviewer 看的参考。

---

## 2. 🟩 OURS — EDAIN CORE (mri_edain_v2/)

我们写的可微归一化层。**完全独立于 nnU-Net**，理论上可以接到任何 PyTorch 分割框架前面。

### modules/

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `modules/rq_spline.py` | ~280 | 🟩 | Rational-quadratic 单调 spline 参数化和 forward；用 26 维 logits → SplineParams |
| `modules/hypernetwork.py` | ~70 | 🟩 | 2-layer GELU MLP, 11→64→64→26, **Fixup 零初始化**最后一层 |
| `modules/percentile.py` | ~95 | 🟩 | 11-landmark γ 计算 (Shah 2011)，支持 quantile 子采样 |
| `modules/standardizer.py` | ~75 | 🟩 | 每个 percentile 维度做 (γ − μ) / σ 标准化 |
| `modules/nyul_init.py` | ~300 | 🔴 | **核心 Nyúl 逻辑**：把训练集的 11 landmark 拟合成 spline anchor θ_0。支持两种 target：`population_nyul`（经典 Nyúl）和 `identity`（退化为 y=x）|
| `modules/edain_layer.py` | ~255 | 🟩 | 顶层 MRIEDAINLayer：拼接 percentile + standardizer + hypernet + spline |
| `modules/foreground.py` | ~50 | 🟩 | 三种 mask 提取方法（nonzero, Otsu, auto）|

### losses/

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `losses/anchor.py` | ~95 | 🟩 | Function-space anchor loss (准则 P3)：50 点网格上 \|\|f_θ − f_θ_0\|\|² |
| `losses/kl.py` | ~140 | 🟩 | KL(X̃ ‖ N(0,1))，soft Gaussian histogram |
| `losses/combined.py` | ~70 | 🟩 | L = L_seg + λ_anc · L_anc + λ_kl · L_kl |

### training/

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `training/precompute.py` | ~360 | 🟩 | 离线计算：对训练集每个 case 算 γ，fit standardizer 和 θ_0，存成 artifact 文件 |
| `training/scheduler.py` | ~90 | 🟩 | 三相 λ 调度（phase 0 frozen / phase 1 anchor strong / phase 2 cosine decay）|
| `training/ema.py` | ~80 | 🟩 | Hypernet EMA shadow（validation 时 swap_in）|

### baselines/

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `baselines/affine_hypernet.py` | ~75 | 🟩 | **P2 kill-switch**：把 spline 退化成 affine 变换。如果它和 EDAIN-identity 表现相同，说明非线性没贡献 |

### tests/

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `tests/test_modules.py` | ~700 | 🟩 | 26 个单元测试，覆盖 spline 单调性、anchor loss、KL、scheduler、precompute |

**EDAIN core 总计：~2700 LOC**

---

## 3. 🟨 OURS — INTEGRATION (edain_nnunet/)

我们写的"把 EDAIN 套进 nnU-Net"的胶水代码。

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `network/edain_wrapper.py` | ~55 | 🟨 | `EDAINWrapper(nn.Module)`：把 MRIEDAINLayer 包到 nnU-Net backbone 外面；通过 case_id 查表使用整卷 γ（修 bug A1）|
| `precompute/nnunet_precompute.py` | ~140 | 🟨 | 直接读 nnU-Net 预处理好的 `.npz`，算 γ + fit θ_0（**避免重新实现 nnU-Net 的预处理**，保证训练/验证 γ 域一致）|
| `trainers/nnUNetTrainerEDAIN.py` | ~120 | 🟨 | **核心 trainer**：继承 `nnUNetTrainer`，只 override 3 个方法：`initialize`、`train_step`、`validation_step`。其他全部从父类继承 |

### scripts/

| 文件 | 标识 | 作用 |
|---|---|---|
| `scripts/train_nnunet_vanilla.sh` | 🟨 | 纯 nnU-Net 对照（reference）|
| `scripts/train_edain_identity_clip.sh` | 🟨 | AB 配置（identity + clip）|
| `scripts/train_edain_identity_noclip.sh` | 🟨 | identity + 无 clip |
| `scripts/train_edain_popnyul_clip.sh` | 🟨 | Plan A（popnyul + clip）|
| `scripts/train_edain_popnyul_noclip.sh` | 🟨 | popnyul + 无 clip |

### docs/

| 文件 | 标识 | 作用 |
|---|---|---|
| `docs/what_changed_vs_nnunet.md` | 🟨 | 详列我们相对 nnU-Net 改了什么/没改什么 |
| `docs/experiment_matrix.md` | 🟨 | 5 个对照实验设计 |

**Integration 总计：~580 LOC + 5 个 .sh + 2 个 docs**

---

## 4. 🟨 OURS — HELPERS (helpers/)

| 文件 | LOC | 标识 | 作用 |
|---|---|---|---|
| `helpers/prepare_nnunet_lipo.py` | ~80 | 🟨 | 把 raw Lipo 数据按 nnU-Net 格式建符号链接到 `$nnUNet_raw/Dataset500_Lipo/` |
| `helpers/make_nnunet_splits.py` | ~50 | 🟨 | 把我们的 `lipo_split.json` 翻译成 nnU-Net 的 `splits_final.json`（保证 5-fold CV 划分一致）|
| `helpers/lipo_split.json` | — | 🟨 | 固定的 5-fold split（114 cases，已排除 Lipo-073 multi-tumor）|

---

## 5. 顶层文件

| 文件 | 标识 | 作用 |
|---|---|---|
| `README.md` | 🟨 | 项目总入口 |
| `LICENSE` | 🟨 | MIT 许可证 + nnU-Net 的 Apache 2.0 通知 |
| `.gitignore` | 🟨 | 标准 Python + nnU-Net 运行时目录排除 |
| `docs/CODE_INVENTORY.md` | 🟨 | **本文档** |
| `docs/REVIEW_CHECKLIST.md` | 🟨 | 给你 review 用的清单 |

---

## 总计

| 归属 | LOC | 文件数 |
|---|---|---|
| 🟦 UPSTREAM nnU-Net 2.7.0 | ~50000 | ~300 |
| 🟩 + 🔴 OURS — EDAIN core | ~2700 | 21 |
| 🟨 OURS — integration + helpers | ~580 + 130 = 710 | 12 + 5 (.sh) |
| **OURS 总计** | **~3400** | **38** |

**Review 范围**：~3400 行我们写的代码，全部不在 `third_party_nnunet/` 下面。
