"""MRI-EDAIN v1 — the 4-sublayer EDAIN layer (Sanna Passino et al. 2024).

Architecture (paper Section 3, Figure 1):

    x  →  h1 (outlier mitigation)  →  h2 (shift)  →  h3 (scale)  →  h4 (power)  →  x_norm

Where:
    h1:  outlier mitigation via tanh winsorization (params alpha, beta)
    h2:  learnable shift (param m)
    h3:  learnable scale (param s)
    h4:  Yeo-Johnson power transform (param lambda)

This package implements the LOCAL-AWARE variant, where shift/scale use
per-image statistics (fg_mean, fg_std) and outlier mitigation centers on
fg_mean. We also wrap the layer with a per-image min-max rescale to [0,1]
BEFORE h1 so that tanh operates in its useful non-linear regime (without this,
tanh saturates on raw MRI values; see lipo_edain_minmax notes).

Used as a learnable replacement for fixed z-score normalization. No
upstream z-score should be applied (run with nnUNet NoNormalization plans).
"""
from .modules.edain_v1_layer import EDAINv1Layer
from .modules.yeo_johnson import yeo_johnson, yeo_johnson_inverse
from .modules.percentile_stats import compute_per_case_stats

__all__ = [
    "EDAINv1Layer",
    "yeo_johnson",
    "yeo_johnson_inverse",
    "compute_per_case_stats",
]
