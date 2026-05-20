# 实验矩阵 (Lipo, 3d_fullres)

## 主对照（fold 0 优先，验证后扩展到 5-fold）

| # | 脚本 | Trainer | EDAIN_ANCHOR_TYPE | EDAIN_OUTLIER_CLIP | 期望 dice (fold 0) |
|---|---|---|---|---|---|
| 1 | `train_nnunet_vanilla.sh` | `nnUNetTrainer` | — | — | **0.825** (reference, 已跑过) |
| 2 | `train_edain_identity_noclip.sh` | `nnUNetTrainerEDAIN` | identity | none | ≥ 0.78 |
| 3 | `train_edain_identity_clip.sh` | `nnUNetTrainerEDAIN` | identity | percentile | **≥ 0.80**（AB 复现） |
| 4 | `train_edain_popnyul_noclip.sh` | `nnUNetTrainerEDAIN` | population_nyul | none | ~0.74 |
| 5 | `train_edain_popnyul_clip.sh` | `nnUNetTrainerEDAIN` | population_nyul | percentile | ~0.66（最差） |

## 关键 sanity check

- **Exp #1 应该等于 `results/lipo_nnunet/` 里 fold 0 的 0.8254**（如果不等，环境/数据有变）
- **Exp #2 应该 ≈ Exp #1**（identity anchor 让 spline 几乎是 identity，hypernet 学到的 Δθ 应该接近 0）
  - 如果 #2 >> #1：hypernet 学到了真有用的偏离 ✓ framework 加分
  - 如果 #2 << #1：framework 有 bug，需要查
- **Exp #3 应该 > Exp #1**（这是论文期望的"AB 在 clean codebase 下更稳"）

## 为什么这是干净的对照

每个 EDAIN 实验和纯 nnU-Net **共享一切**（augmentation, optimizer, data split, validation protocol, network arch）。dice 差异只能来自 EDAIN 层。

## 跑的顺序建议

1. **首先**：跑 Exp #1 fold 0，确认能复现 0.8254（≤ 2 天）
2. **同时**：跑 Exp #2 fold 0（identity + noclip），确认 framework 不伤害（≤ 2 天）
3. **然后**：跑 Exp #3 fold 0（AB 复现，~期望 0.80+），验证主要结论（≤ 2 天）
4. **最后**：如果 #3 复现成功，把 #3 扩展到 fold 1-4 做完整 5-fold CV（~10 天）
5. **Plan A 验证**：跑 #4 和 #5 fold 0，确认 popnyul + clip 真的最差
