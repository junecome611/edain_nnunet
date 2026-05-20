"""Yeo-Johnson power transformation (h4 sublayer of EDAIN).

Paper Equation (3):

    YJ(x; lambda) =
        ( (x+1)^lambda - 1 ) / lambda          if lambda != 0, x >= 0
        log(x + 1)                              if lambda == 0, x >= 0
        ( (1-x)^(2-lambda) - 1 ) / (lambda-2)  if lambda != 2, x < 0
        -log(1 - x)                             if lambda == 2, x < 0

Properties:
    - Monotonic for all lambda
    - At lambda = 1, YJ(x; 1) = x (identity)         => safe init
    - Defined for x in R (handles negative values, unlike Box-Cox)
    - Differentiable in lambda and x except at the singular cases above

For backward stability we always evaluate the (lambda != 0, lambda != 2)
branches with the protections below:

    1. Near lambda = 0 (|lambda| < eps):
         use the log branch directly (smooth limit)
    2. Near lambda = 2 (|lambda - 2| < eps):
         use the -log branch directly
    3. (x + 1) and (1 - x) are clamped to >= 1e-6 to avoid 0^p when x = -1 or x = 1

This keeps lambda fully differentiable.
"""
from __future__ import annotations
import torch


_LAMBDA_EPS = 1e-3   # switch to log-branch when |lambda| < eps or |lambda-2| < eps
_BASE_EPS   = 1e-6   # clamp on (x+1) and (1-x)


def yeo_johnson(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Forward Yeo-Johnson transform, element-wise.

    Args:
        x:   tensor of any shape, real-valued.
        lam: scalar tensor (single global lambda) or broadcastable to x.

    Returns:
        Tensor of the same shape as x.
    """
    # Positive branch: x >= 0
    pos_mask = x >= 0
    # For x >= 0 we use (x+1)^lambda
    base_pos = torch.clamp(x + 1.0, min=_BASE_EPS)

    if torch.abs(lam) < _LAMBDA_EPS:
        # lambda ~ 0: use limit log(x+1)
        y_pos = torch.log(base_pos)
    else:
        y_pos = (torch.pow(base_pos, lam) - 1.0) / lam

    # Negative branch: x < 0
    base_neg = torch.clamp(1.0 - x, min=_BASE_EPS)
    if torch.abs(lam - 2.0) < _LAMBDA_EPS:
        # lambda ~ 2: use limit -log(1-x)
        y_neg = -torch.log(base_neg)
    else:
        y_neg = -(torch.pow(base_neg, 2.0 - lam) - 1.0) / (2.0 - lam)
        # Paper Eq 3 third row: ((1-x)^(2-lambda) - 1) / (lambda - 2).
        # That equals -((1-x)^(2-lambda) - 1) / (2 - lambda) which is what we
        # compute above; we use the (2-lambda) form for slightly nicer gradients.

    return torch.where(pos_mask, y_pos, y_neg)


def yeo_johnson_inverse(y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Inverse Yeo-Johnson transform. Useful only for EDAIN-KL or diagnostics.

    Args:
        y:   tensor of any shape.
        lam: scalar tensor.
    """
    pos_mask = y >= 0
    # Positive: (y*lam + 1)^(1/lam) - 1
    if torch.abs(lam) < _LAMBDA_EPS:
        x_pos = torch.exp(y) - 1.0
    else:
        x_pos = torch.pow(torch.clamp(y * lam + 1.0, min=_BASE_EPS), 1.0 / lam) - 1.0

    # Negative
    if torch.abs(lam - 2.0) < _LAMBDA_EPS:
        x_neg = 1.0 - torch.exp(-y)
    else:
        x_neg = 1.0 - torch.pow(torch.clamp(-y * (2.0 - lam) + 1.0, min=_BASE_EPS),
                                 1.0 / (2.0 - lam))

    return torch.where(pos_mask, x_pos, x_neg)
