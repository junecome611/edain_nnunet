# 我们的 EDAIN 集成相对 nnU-Net v2 改了什么

> 这个文档列出 `edain_nnunet/` 相对 upstream nnU-Net v2 (`nnunet/nnUNet/nnunetv2`) 的**全部**差异。任何不在这里列的东西都和 nnU-Net 完全一致。

---

## 1. 改的只有 3 件事

| # | 位置 | 改了什么 | 行数 |
|---|---|---|---|
| 1 | `network/edain_wrapper.py` | 新增 `EDAINWrapper(nn.Module)`：在 nnU-Net backbone 前接一个 MRIEDAINLayer | ~50 |
| 2 | `precompute/nnunet_precompute.py` | 新增：在 nnU-Net 已预处理的 `.npz` 上算 γ + fit standardizer/θ_0 | ~120 |
| 3 | `trainers/nnUNetTrainerEDAIN.py` | 继承 `nnUNetTrainer`，override 3 个方法：`initialize`, `train_step`, `validation_step` | ~120 |

**总新增代码 ~300 行，零行修改 upstream nnU-Net**。

---

## 2. 没改的东西（即从 nnU-Net 完全继承）

| 组件 | nnU-Net 源 |
|---|---|
| 数据增广（9 种增广 + 概率）| [nnUNetTrainer.py:739-889](../../nnunet/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) `get_training_transforms` |
| Z-score 预处理 | [default_normalization_schemes.py:27](../../nnunet/nnUNet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py) `ZScoreNormalization` |
| 训练循环（AMP, grad clip 12, etc.）| [nnUNetTrainer.py:1019](../../nnunet/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) `train_step` |
| 验证循环 | [nnUNetTrainer.py:1066](../../nnunet/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) `validation_step` |
| Optimizer (SGD nesterov 0.99, wd 3e-5) | [nnUNetTrainer.py:552](../../nnunet/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) `configure_optimizers` |
| Loss (DC + CE 组合) | nnUNet 的 `_build_loss()` |
| LR schedule (PolyLR) | nnUNet 的 `configure_optimizers` |
| Deep supervision (5 levels) | nnUNet network arch (`PlainConvUNet`) |
| Patch size [32, 224, 256] | nnUNetPlans 自动生成 |
| Sliding window 推理 | nnUNet inference 模块 |
| Mirror axes / TTA | nnUNet 默认 |
| 5-fold CV split | 通过 `code/make_nnunet_splits.py` 注入 `splits_final.json`，**完全相同** |

---

## 3. EDAIN 集成的 3 个 hook 详解

### Hook 1：`initialize()`

**位置**：[trainers/nnUNetTrainerEDAIN.py](../trainers/nnUNetTrainerEDAIN.py)

```python
def initialize(self):
    super().initialize()                       # ← 让 nnU-Net 完成所有标准 init
    artifacts = self._precompute_edain()       # ← 算 γ + fit θ_0
    standardizer = CoordinateStandardizer(...)
    edain = MRIEDAINLayer(standardizer, artifacts['theta_0'], ...)
    self.network = EDAINWrapper(               # ← 把 backbone 包起来
        edain, self.network, artifacts['case_gammas']
    ).to(self.device)
    self.optimizer, self.lr_scheduler = self.configure_optimizers()  # ← 重建
```

**作用**：在 nnU-Net 标准 init 之后，注入 EDAIN 层。**优化器必须重建**因为 `self.network` 变了（新增了 EDAIN 的 hypernet 参数）。

### Hook 2：`train_step()` / `validation_step()`

**位置**：[trainers/nnUNetTrainerEDAIN.py](../trainers/nnUNetTrainerEDAIN.py)

```python
def train_step(self, batch):
    case_ids = batch.get("keys", None)
    self.network.set_current_batch(case_ids)   # ← 把 case_id 传给 wrapper
    return super().train_step(batch)           # ← 其他完全继承
```

**作用**：每个 batch forward 之前把 case_ids stash 到 wrapper 上。Wrapper 用这个查 precomputed whole-volume γ（这就是 bug A1 的修复）。

---

## 4. Bug A1/A2 的修复方式

### A1：训练/验证 γ 域不一致 ✅ 已修复

**旧方案**：训练时每个 patch 重新算 γ → 和验证时整卷 γ 分布不一致
**新方案**：精算阶段对每个 case 算一次**整卷 γ**，存进 `artifacts['case_gammas']`。Wrapper.forward 通过 case_id 查表，**训练和验证用同一个 γ**。
**关键代码**：`EDAINWrapper._lookup_gammas()` ([network/edain_wrapper.py:50](../network/edain_wrapper.py))

### A2：mask 在增广后污染 ✅ 自动消失

**旧方案**：自写 MONAI augmentation，`mask = (X != 0)` 在 NnUNetRandGammaD 等之后才算，背景被污染
**新方案**：用 nnU-Net 真正的 `GammaTransform`（带 `p_retain_stats=1` 保持均值方差不变），不污染 mask
**关键**：完全不需要我们做任何事——直接继承 nnU-Net 的 `get_training_transforms` 就解决了

---

## 5. 怎么验证"我们的差异真的只有 EDAIN"

跑两个对照：

```bash
# 纯 nnU-Net
nnUNetv2_train 500 3d_fullres 0 --npz

# EDAIN 包了 nnU-Net (anchor=identity 让 spline 起点≈无操作)
EDAIN_ANCHOR_TYPE=identity EDAIN_OUTLIER_CLIP=none \
    nnUNetv2_train 500 3d_fullres 0 -tr nnUNetTrainerEDAIN --npz
```

- 如果两者 dice 几乎一致 → EDAIN-identity 在 hypernet 学到 0 偏离时**严格退化为 nnU-Net**（P1 非吸收必须实证的实证版本）
- 如果 EDAIN-identity > vanilla nnU-Net → hypernet 学到了真有用的偏离
- 如果 EDAIN-identity < vanilla nnU-Net → 框架本身有伤害（需要继续 debug）

---

## 6. 为什么这个集成方案是"对的"

1. **科学**：相对 nnU-Net 的差异**只有 EDAIN 层**，dice 差异 100% 可归因
2. **工程**：300 行新代码 vs 重写 1000+ 行 MONAI pipeline，bug 面积小一个量级
3. **可比性**：直接和发布过的 nnU-Net 性能（reference 0.7790 ± 0.034）对比
4. **可维护**：upstream nnU-Net 升级时我们只需要确认 `nnUNetTrainer` 三个被 override 的方法签名没变
