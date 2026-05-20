"""edain_nnunet: MRI-EDAIN v2 layer integrated with the upstream nnU-Net v2
training framework.

The whole point of this package is that it does NOT modify nnU-Net itself.
We register a custom trainer (nnUNetTrainerEDAIN) which inherits from
nnUNetTrainer and only overrides initialize() + the two step methods.

Usage (cluster):
    export PYTHONPATH=/path/to/Adaptive_Preprocessing:$PYTHONPATH
    export EDAIN_ANCHOR_TYPE=identity
    export EDAIN_OUTLIER_CLIP=percentile
    nnUNetv2_train 500 3d_fullres 0 -tr nnUNetTrainerEDAIN \\
        -p nnUNetPlans --npz
"""
