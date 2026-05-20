"""nnUNetTrainerEDAINv1Power — EDAIN v1 with all 4 sublayers (h1+h2+h3+h4).

This is exactly the same as nnUNetTrainerEDAINv1 except it auto-enables the
power transform (h4, Yeo-Johnson) by overriding the env-var read. We keep it
as a subclass so the experiment matrix can pick it via its class name.

Compared to nnUNetTrainerEDAINv1 (without h4), the only differences are:
  - h4 is active: x4 = YJ(x3; lambda)
  - lambda is a learnable parameter (init 1.0 = identity)
  - lambda has its own LR multiplier (default 10)

Configuration / env vars: same as nnUNetTrainerEDAINv1, but EDAIN_V1_USE_POWER
is forced to True.
"""
from __future__ import annotations
from .nnUNetTrainerEDAINv1 import nnUNetTrainerEDAINv1


class nnUNetTrainerEDAINv1Power(nnUNetTrainerEDAINv1):
    """EDAIN v1 with h4 (Yeo-Johnson power transform) always on."""

    @property
    def use_power_transform(self) -> bool:
        return True
