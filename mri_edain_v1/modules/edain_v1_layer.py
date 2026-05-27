"""EDAIN v1 layer (4 sublayers, local-aware z-score formulation).

Faithful port of `results/lipo_v3/lipo_edain_zscore_3_nointensityaug.py` —
the team's prior working z-score EDAIN on Lipo. No pre-rescale: the layer
runs directly on raw MRI intensities and uses (fg_mean, fg_std) per case
as the reference scale for OM / shift / scale.

Architecture (forward order on raw x):

    raw x ─▶ h1 (outlier mit, local-aware: μ̂ = fg_mean, β raw units)
          ─▶ h2 (shift, RELATIVE: x_om - m·fg_mean)
          ─▶ h3 (scale, RELATIVE: / (s · fg_std))
          ─▶ h4 (Yeo-Johnson power, λ — OPTIONAL)
          ─▶ x_norm

At α=1, m=1, s=1 the layer reduces to winsorized z-score:
    x_norm = (β·tanh((x-μ)/β)) / σ
which is the natural "z-score with outlier handling" starting point.

Parameters (all global learnable scalars):
    alpha   in [0, 1]            sigmoid(_alpha_raw)         init 0.5
    beta    in [beta_min, inf)   softplus(_beta_raw)+β_min   init β_init (raw units!)
    m       in R                 plain nn.Parameter          init 1.0
    s       in (s_min, inf)      softplus(_s_raw)+s_min      init 1.0
    lambda  in R                 plain nn.Parameter (h4 on)  init 1.0

IMPORTANT — β has raw-intensity units (not dimensionless).
    The sublayer is β·tanh((x-μ)/β) + μ, so (x-μ)/β must be dimensionless.
    On Lipo, fg_std median ~ 270 (raw units). Pick β_init in raw units —
    typical sensible default ~ 3·fg_std_median ≈ 800, which puts the tanh
    transition at roughly 3σ (mild winsorization on outliers, near no-op
    for the central distribution). Set via env var EDAIN_V1_INIT_BETA.

Forward expects:
    x:        (B, 1, D, H, W) raw MRI patch (post-augmentation, NOT z-scored)
    fg_stats: (B, 4) tensor with [fg_mean, fg_std, fg_p2, fg_p98] per sample.
              fg_p2/fg_p98 are tolerated but UNUSED in z-score form.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yeo_johnson import yeo_johnson


class EDAINv1Layer(nn.Module):
    """4-sublayer EDAIN, local-aware z-score formulation."""

    def __init__(
        self,
        init_alpha: float = 0.5,
        init_beta: float = 800.0,    # raw-intensity units; ~3·median(fg_std) on Lipo
        init_m: float = 1.0,         # relative coefficient (× fg_mean), 1.0 = full shift
        init_s: float = 1.0,         # relative coefficient (× fg_std),  1.0 = full scale
        init_lambda: float = 1.0,
        beta_min: float = 1.0,
        s_min: float = 1e-3,
        use_power_transform: bool = False,
    ):
        super().__init__()
        self.beta_min = float(beta_min)
        self.s_min = float(s_min)
        self.use_power_transform = bool(use_power_transform)

        # alpha: sigmoid(_alpha_raw) -> [0, 1]
        self._alpha_raw = nn.Parameter(torch.tensor(
            self._inverse_sigmoid(init_alpha), dtype=torch.float32))

        # beta: softplus(_beta_raw) + beta_min -> [beta_min, inf)
        self._beta_raw = nn.Parameter(torch.tensor(
            self._inverse_softplus(max(init_beta - beta_min, 1e-3)),
            dtype=torch.float32))

        # m: unconstrained scalar; multiplied by fg_mean during forward
        self.m = nn.Parameter(torch.tensor(float(init_m), dtype=torch.float32))

        # s: softplus(_s_raw) + s_min -> (s_min, inf); multiplied by fg_std
        self._s_raw = nn.Parameter(torch.tensor(
            self._inverse_softplus(max(init_s - s_min, 1e-3)),
            dtype=torch.float32))

        # lambda: unconstrained, only used when use_power_transform
        if self.use_power_transform:
            self.lambda_param = nn.Parameter(torch.tensor(
                float(init_lambda), dtype=torch.float32))
        else:
            # Register as buffer for consistent state_dict regardless of mode
            self.register_buffer(
                "lambda_param", torch.tensor(float(init_lambda), dtype=torch.float32))

    # ----- constrained accessors -----
    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw) + self.beta_min

    @property
    def s(self) -> torch.Tensor:
        return F.softplus(self._s_raw) + self.s_min

    @staticmethod
    def _inverse_sigmoid(y: float) -> float:
        y = float(np.clip(y, 1e-6, 1 - 1e-6))
        return float(np.log(y / (1 - y)))

    @staticmethod
    def _inverse_softplus(y: float) -> float:
        y = float(max(y, 1e-6))
        if y > 20:
            return y
        return float(np.log(np.exp(y) - 1))

    # ----- learnable parameter dict for per-sublayer LR setup -----
    def named_sublayer_params(self):
        """Yield (sublayer_name, parameter) for each learnable EDAIN parameter.

        Used by EDAINv1Wrapper to set up per-sublayer learning rate groups.
        """
        yield "alpha", self._alpha_raw
        yield "beta", self._beta_raw
        yield "shift", self.m
        yield "scale", self._s_raw
        if self.use_power_transform:
            yield "power", self.lambda_param

    # ----- forward -----
    def forward(self, x: torch.Tensor, fg_stats: torch.Tensor) -> torch.Tensor:
        """Apply EDAIN v1 transform (z-score form) on a batch of raw images.

        Args:
            x:        (B, 1, D, H, W) raw MRI values.
            fg_stats: (B, 4) tensor [fg_mean, fg_std, fg_p2, fg_p98] per sample.
                      Only fg_mean and fg_std are used; p2/p98 are ignored.
                      If B_stats < B (sliding-window inference), broadcast.

        Returns:
            x_norm: same shape as x, normalized.
        """
        B = x.shape[0]
        if fg_stats.dim() == 1:
            fg_stats = fg_stats.unsqueeze(0)
        if fg_stats.shape[0] != B:
            fg_stats = fg_stats[:1].expand(B, -1)

        # Fresh contiguous tensor on x's device/dtype. .contiguous() defends
        # against expanded tensors with stride-0 on dim 0 (would crash any
        # downstream in-place op).
        fg_stats = fg_stats.to(device=x.device, dtype=x.dtype).contiguous()

        view_shape = (B, 1, 1, 1, 1)
        fg_mean = fg_stats[:, 0].reshape(view_shape)
        fg_std  = fg_stats[:, 1].reshape(view_shape).clamp(min=1e-6)
        # fg_p2 / fg_p98 (indices 2, 3) are unused in z-score form.

        alpha = self.alpha
        beta = self.beta

        # ---- h1: outlier mitigation, local-aware (μ̂ = fg_mean) ----
        # β has raw-intensity units; (x - fg_mean)/β is dimensionless.
        x_centered = x - fg_mean
        x_w = beta * torch.tanh(x_centered / beta) + fg_mean
        x_om = alpha * x_w + (1.0 - alpha) * x

        # ---- h2: shift, RELATIVE to fg_mean ----
        # At m=1, this fully subtracts fg_mean (z-score centering).
        x_shifted = x_om - self.m * fg_mean

        # ---- h3: scale, RELATIVE to fg_std ----
        # At s=1, this fully divides by fg_std (z-score scaling).
        x_scaled = x_shifted / (self.s * fg_std)

        # ---- h4: power transform (optional) ----
        if self.use_power_transform:
            x_out = yeo_johnson(x_scaled, self.lambda_param)
        else:
            x_out = x_scaled

        return x_out

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (
                f"alpha={self.alpha.item():.4f}, "
                f"beta={self.beta.item():.4f}, "
                f"m={self.m.item():.4f}, "
                f"s={self.s.item():.4f}, "
                f"lambda={self.lambda_param.item():.4f}, "
                f"use_power={self.use_power_transform}"
            )
