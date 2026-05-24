"""EDAIN v1 layer (4 sublayers, local-aware, with optional power transform).

Architecture (paper Section 3, with our min-max prelude):

    x ─▶ [optional pre-rescale to ~[0,1] via fg_p2/fg_p98]
       ─▶ h1 (outlier mit, alpha, beta, tanh winsorize)
       ─▶ h2 (shift,  m)
       ─▶ h3 (scale,  s)
       ─▶ h4 (power transform, lambda  — OPTIONAL)
       ─▶ x_norm

Why the pre-rescale?
    The raw Lipo MRI intensity range is 0..thousands, but EDAIN's β has to live
    in the same scale as |x - μ̂|. With raw input you need β ≈ 1300 to keep
    tanh in its useful regime — too brittle (sensitive to extreme cases).
    Following the team's prior work on `lipo_edain_minmax_1.py`, we add a
    per-image robust rescale to [0,1] using (fg_p2, fg_p98) FIRST, then
    EDAIN parameters live in their natural [0,1] scale (β ≈ 1.5, m ≈ 0).

    This rescale itself is FIXED (uses precomputed percentiles, not learned).
    EDAIN still does the learnable normalization on top.

Parameters (all global learnable scalars, one set per training run):
    alpha   in [0, 1]            sigmoid(_alpha_raw)
    beta    in [beta_min, inf)   softplus(_beta_raw) + beta_min
    m       in R                 plain nn.Parameter
    s       in (0, inf)          softplus(_s_raw) + eps
    lambda  in R                 plain nn.Parameter (only if use_power=True)

Initial values (verified on raw Lipo by notebooks/edain_init_analysis/):
    alpha = 0.5    moderate winsorize blend at init (paper default)
    beta  = 1.5    on rescaled [0,1] input, tanh transitions around |x_norm|=1.5
    m     = 0.0    no extra shift after rescale (output starts centered at 0.5)
    s     = 1.0    no extra scale (h3 is identity at init)
    lambda= 1.0    YJ(x; 1) = x identity (h4 is no-op at init when enabled)

Forward expects:
    x:        (B, 1, D, H, W) raw MRI patch (post-augmentation, NOT z-scored)
    fg_stats: (B, 4) tensor with [fg_mean, fg_std, fg_p2, fg_p98] per sample
              (provided by EDAINv1Wrapper via case_id lookup)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yeo_johnson import yeo_johnson


class EDAINv1Layer(nn.Module):
    """4-sublayer EDAIN (Sanna Passino et al. 2024), local-aware."""

    def __init__(
        self,
        init_alpha: float = 0.5,
        init_beta: float = 1.5,
        init_m: float = 0.0,
        init_s: float = 1.0,
        init_lambda: float = 1.0,
        beta_min: float = 0.1,
        s_min: float = 1e-3,
        use_power_transform: bool = False,
        rescale_with_percentile: bool = True,
    ):
        super().__init__()
        self.beta_min = float(beta_min)
        self.s_min = float(s_min)
        self.use_power_transform = bool(use_power_transform)
        self.rescale_with_percentile = bool(rescale_with_percentile)

        # alpha: sigmoid(_alpha_raw) -> [0, 1]
        self._alpha_raw = nn.Parameter(torch.tensor(
            self._inverse_sigmoid(init_alpha), dtype=torch.float32))

        # beta: softplus(_beta_raw) + beta_min -> [beta_min, inf)
        self._beta_raw = nn.Parameter(torch.tensor(
            self._inverse_softplus(max(init_beta - beta_min, 1e-3)),
            dtype=torch.float32))

        # m: unconstrained
        self.m = nn.Parameter(torch.tensor(float(init_m), dtype=torch.float32))

        # s: softplus(_s_raw) + s_min -> (s_min, inf)
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
        """Apply EDAIN v1 transform on a batch of raw images.

        Args:
            x:        (B, 1, D, H, W) raw MRI values.
            fg_stats: (B, 4) tensor [fg_mean, fg_std, fg_p2, fg_p98] per sample.
                      If B_stats < B (sliding-window inference), broadcast.

        Returns:
            x_norm: same shape as x, normalized.
        """
        B = x.shape[0]
        # Broadcast fg_stats if needed (sliding-window inference reuses 1 stats row)
        if fg_stats.dim() == 1:
            fg_stats = fg_stats.unsqueeze(0)
        if fg_stats.shape[0] != B:
            fg_stats = fg_stats[:1].expand(B, -1)

        # Normalize fg_stats to a fresh contiguous (B, 4) tensor on x's
        # device/dtype.
        #   - .to(...) handles device + dtype in one go.
        #   - .contiguous() defends against the case where fg_stats came in as
        #     an expanded tensor (stride 0 on dim 0). Without this, the in-place
        #     clamp_ below would crash on "more than one element of the
        #     written-to tensor refers to a single memory location."
        fg_stats = fg_stats.to(device=x.device, dtype=x.dtype).contiguous()

        # (B, 4) -> 4 broadcastable scalars of shape (B, 1, 1, 1, 1)
        # Use .reshape() (handles non-contig) and .clamp() (not in-place).
        view_shape = (B, 1, 1, 1, 1)
        fg_mean = fg_stats[:, 0].reshape(view_shape)
        fg_std = fg_stats[:, 1].reshape(view_shape).clamp(min=1e-6)
        fg_p2 = fg_stats[:, 2].reshape(view_shape)
        fg_p98 = fg_stats[:, 3].reshape(view_shape)

        # ---- Optional pre-rescale to ~[0,1] via (p2, p98) ----
        # This is a FIXED transform (not learnable) that gives β a sensible scale.
        if self.rescale_with_percentile:
            range_p = (fg_p98 - fg_p2).clamp(min=1e-6)
            x_in = (x - fg_p2) / range_p          # ~[0, 1] for the central 96%
            mu_hat = (fg_mean - fg_p2) / range_p  # rescaled mu; usually ~0.4
        else:
            x_in = x
            mu_hat = fg_mean

        # ---- h1: outlier mitigation (tanh winsorization) ----
        alpha = self.alpha
        beta = self.beta
        x_centered = x_in - mu_hat
        x_w = beta * torch.tanh(x_centered / beta) + mu_hat
        x_om = alpha * x_w + (1.0 - alpha) * x_in

        # ---- h2: shift   h3: scale ----
        # Combined: (x_om - m) / s
        x_ss = (x_om - self.m) / self.s

        # ---- h4: power transform (optional) ----
        if self.use_power_transform:
            x_out = yeo_johnson(x_ss, self.lambda_param)
        else:
            x_out = x_ss

        return x_out

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (
                f"alpha={self.alpha.item():.4f}, "
                f"beta={self.beta.item():.4f}, "
                f"m={self.m.item():.4f}, "
                f"s={self.s.item():.4f}, "
                f"lambda={self.lambda_param.item():.4f}, "
                f"use_power={self.use_power_transform}, "
                f"rescale_with_percentile={self.rescale_with_percentile}"
            )
