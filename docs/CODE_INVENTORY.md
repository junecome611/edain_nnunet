# 代码清单（Code Inventory）

> 这个文档列出仓库内**每一个文件**的归属、来源和作用。
> 用 🟦 / 🔵 / 🟣 / 🟨 标识归属。

## 归属图例

| 标识 | 含义 | License |
|---|---|---|
| 🟦 UPSTREAM | nnU-Net 2.7.0 原版（未修改）| Apache 2.0 |
| 🔵 OURS — EDAIN | EDAIN v1 实现（4 sublayer，paper Sanna Passino 2024）| MIT |
| 🟣 OURS — Nyul | Nyul 风格 spline + hypernet（我们的 v2）| MIT |
| 🟨 OURS — Integration / helpers | nnU-Net 集成胶水代码 + 数据准备 | MIT |

---

## 1. 🟦 UPSTREAM (`third_party_nnunet/`)

整个 `third_party_nnunet/` 是 [MIC-DKFZ/nnUNet v2.7.0](https://github.com/MIC-DKFZ/nnUNet) 未修改的副本。**修改总数：0 行**。

我们用 `pip install nnunetv2==2.7.0` 在 cluster 上运行（见 `requirements.txt`），仓库里保留这份副本是为了让 reviewer 能确认版本和 license。

---

## 2. 🔵 OURS — EDAIN v1 (`mri_edain_v1/`)

实现 Sanna Passino et al. 2024 的 **4 sublayer EDAIN**：

| 文件 | 标识 | 作用 |
|---|---|---|
| `modules/edain_v1_layer.py` | 🔵 | `EDAINv1Layer`：完整 4-sublayer 前向，含 per-sublayer 参数访问器 |
| `modules/yeo_johnson.py` | 🔵 | h4 power transform 的可微 Yeo-Johnson 实现 |
| `modules/percentile_stats.py` | 🔵 | per-case 统计量计算（fg_mean / fg_std / fg_p2 / fg_p98）+ JSON 缓存读写 |
| `precompute/precompute_v1_stats.py` | 🔵 | 离线脚本，在 nnU-Net 已预处理的 raw `.npz` 上算上述统计量 |

### EDAIN v1 数学（对应论文公式）

```
x_norm  = (x - fg_p2) / (fg_p98 - fg_p2)          [optional pre-rescale to ~[0,1]]
mu_hat  = (fg_mean - fg_p2) / (fg_p98 - fg_p2)
h1(x)   = alpha * (beta * tanh((x_norm - mu_hat)/beta) + mu_hat)
            + (1 - alpha) * x_norm                       [outlier mit]
h2(h1)  = h1 - m                                          [shift]
h3(h2)  = h2 / s                                          [scale]
h4(h3)  = YeoJohnson(h3; lambda)                          [optional power transform]
```

可学习参数：α ∈ [0,1], β ∈ [β_min, ∞), m ∈ ℝ, s ∈ (0, ∞), λ ∈ ℝ。

---

## 3. 🟣 OURS — Nyul (`mri_edain_v2/`)

实现我们的 **input-conditional 单调 RQ-spline 归一化**：

| 文件 | 标识 | 作用 |
|---|---|---|
| `modules/rq_spline.py` | 🟣 | RQ-spline 参数化与 forward（Durkan 2019, NeurIPS）|
| `modules/hypernetwork.py` | 🟣 | 2 层 GELU MLP，11 → 64 → 64 → 26，Fixup 零初始化 |
| `modules/nyul_init.py` | 🟣 | **核心 Nyul anchor 拟合**（population_nyul + identity）|
| `modules/edain_layer.py` | 🟣 | `MRIEDAINLayer`：拼接 percentile + standardizer + hypernet + spline |
| `modules/percentile.py` | 🟣 | 11-landmark γ 计算（Shah 2011）|
| `modules/standardizer.py` | 🟣 | per-coordinate γ 标准化 |
| `modules/foreground.py` | 🟣 | mask 提取（nonzero / Otsu / auto）|
| `losses/anchor.py` | 🟣 | Function-space anchor loss |
| `losses/kl.py` | 🟣 | KL → N(0,1) 弱正则 |
| `losses/combined.py` | 🟣 | 三项加权组合 |
| `training/precompute.py` | 🟣 | 离线拟合 standardizer + θ_0 |
| `training/scheduler.py` | 🟣 | 三相 λ 调度 |
| `training/ema.py` | 🟣 | hypernet EMA shadow |
| `baselines/affine_hypernet.py` | 🟣 | P2 kill-switch |
| `tests/test_modules.py` | 🟣 | 26 个单元测试 |

---

## 4. 🟨 INTEGRATION (`edain_nnunet/`)

| 文件 | 标识 | 作用 |
|---|---|---|
| `trainers/nnUNetTrainerEDAINv1.py` | 🟨 | EDAIN v1（不含 h4）trainer |
| `trainers/nnUNetTrainerEDAINv1Power.py` | 🟨 | EDAIN v1 + h4 trainer（子类继承） |
| `trainers/nnUNetTrainerNyul.py` | 🟨 | Nyul 风格 spline trainer |
| `network/edain_v1_wrapper.py` | 🟨 | EDAIN v1 + nnU-Net backbone 的 wrapper |
| `network/edain_wrapper.py` | 🟨 | Nyul + nnU-Net backbone 的 wrapper |
| `precompute/nnunet_precompute.py` | 🟨 | Nyul 离线 precompute（读 z-scored .npz）|
| `plans/make_raw_plans.py` | 🟨 | **关键**：生成 `nnUNetPlans_raw`（NoNormalization），让 EDAIN v1 能看到 raw |

---

## 5. 🟨 HELPERS (`helpers/`)

| 文件 | 标识 | 作用 |
|---|---|---|
| `prepare_nnunet_lipo.py` | 🟨 | 把 raw Lipo 数据按 nnU-Net 格式建符号链接 |
| `make_nnunet_splits.py` | 🟨 | 把 `lipo_split.json` 翻译成 nnU-Net 的 `splits_final.json` |
| `lipo_split.json` | 🟨 | 固定的 5-fold split（114 cases，已排除 Lipo-073）|

---

## 6. 🟨 SCRIPTS (`scripts/`)

| 文件 | 标识 | 作用 |
|---|---|---|
| `00_setup_data.sh` | 🟨 | 一次性 setup：数据准备 + 两种 preprocessing + precompute |
| `01_baseline_nnunet.sh` | 🟨 | EXP 0: 纯 nnU-Net |
| `02_edain_v1.sh` | 🟨 | EXP 1: EDAIN v1 (h1+h2+h3) |
| `03_edain_v1_power.sh` | 🟨 | EXP 2: EDAIN v1 + h4 |
| `04_nyul_identity.sh` | 🟨 | EXP 3: Nyul identity anchor |
| `05_nyul_popnyul.sh` | 🟨 | EXP 4: Nyul popnyul anchor |

---

## 7. 顶层文件

| 文件 | 标识 |
|---|---|
| `README.md` | 🟨 |
| `LICENSE` | 🟨 (MIT) + nnU-Net Apache 2.0 notice |
| `.gitignore` | 🟨 |
| `requirements.txt` | 🟨 |
| `docs/CODE_INVENTORY.md` | 🟨 (本文档) |
| `docs/REVIEW_CHECKLIST.md` | 🟨 |

---

## 总览统计

| 归属 | LOC | 文件数 |
|---|---|---|
| 🟦 nnU-Net 2.7.0 | ~50000 | ~300 |
| 🔵 EDAIN v1 | ~500 | 5 |
| 🟣 Nyul | ~2700 | 21 |
| 🟨 集成 + scripts + helpers | ~1100 | 22 |
| **OURS total** | **~4300** | **48** |

Review 范围：~4300 行我们写的代码，全部不在 `third_party_nnunet/` 下面。
