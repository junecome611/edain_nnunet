# edain_nnunet/

MRI-EDAIN v2 layer integrated with the upstream nnU-Net v2 training framework.

## 设计原则

- **不动 upstream nnU-Net 一行** (`nnunet/nnUNet/` 完全只读)
- **不动 EDAIN core modules 一行** (`mri_edain_v2/` 完全只读)
- 所有差异集中在这个文件夹的 **3 个核心文件 + 一些脚本**

## 结构

```
edain_nnunet/
├── network/edain_wrapper.py        # 把 EDAIN 包到 nnU-Net 网络外面
├── precompute/nnunet_precompute.py # 在 nnU-Net 预处理 .npz 上 fit θ_0
├── trainers/nnUNetTrainerEDAIN.py  # 继承 nnUNetTrainer + 3 个 override
├── scripts/*.sh                    # Slurm 任务（每个配置一份）
└── docs/what_changed_vs_nnunet.md  # 详细差异列表
```

## 实验矩阵

| 脚本 | anchor | clip | 用 trainer | 期望 |
|---|---|---|---|---|
| `train_nnunet_vanilla.sh` | — | — | `nnUNetTrainer` | nnU-Net 0.825 reference |
| `train_edain_identity_noclip.sh` | identity | none | `nnUNetTrainerEDAIN` | 框架退化测试 |
| `train_edain_identity_clip.sh` | identity | percentile | `nnUNetTrainerEDAIN` | **AB 配置复现** (旧 0.8033) |
| `train_edain_popnyul_noclip.sh` | popnyul | none | `nnUNetTrainerEDAIN` | 旧 0.7389 |
| `train_edain_popnyul_clip.sh` | popnyul | percentile | `nnUNetTrainerEDAIN` | 旧 0.6566（最差） |

## 用法

每个 .sh 接受 `FOLD` 环境变量（默认 0）：

```bash
sbatch --export=FOLD=0 scripts/train_edain_identity_clip.sh
sbatch --export=FOLD=1 scripts/train_edain_identity_clip.sh
# ...
```

## 详见

- [docs/what_changed_vs_nnunet.md](docs/what_changed_vs_nnunet.md): upstream 差异列表
- [docs/experiment_matrix.md](docs/experiment_matrix.md): 实验设计
- 旧实验回顾: `../conversation_summary_mri_edain_v2_session_v2.md`
