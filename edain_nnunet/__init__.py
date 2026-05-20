"""edain_nnunet: integration glue between EDAIN v1 / Nyul and upstream nnU-Net v2.

This package does NOT modify nnU-Net. It registers custom trainer subclasses:

    nnUNetTrainerEDAINv1        - EDAIN v1 (paper, 4 sublayers)
    nnUNetTrainerEDAINv1Power   - EDAIN v1 with Yeo-Johnson power transform
    nnUNetTrainerNyul           - Nyul-inspired RQ-spline + hypernet (our v2)

For details on each design see README.md.

Usage (cluster):
    export PYTHONPATH=/path/to/edain_nnunet:$PYTHONPATH
    sbatch scripts/02_edain_v1.sh
    # or for Nyul:
    sbatch scripts/04_nyul_identity.sh
"""
