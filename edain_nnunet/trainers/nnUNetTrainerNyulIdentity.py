"""nnUNetTrainerNyulIdentity — Nyul spline trainer with identity anchor + clip.

Thin subclass of nnUNetTrainerNyul that fixes the configuration:
    anchor_type    = "identity"
    outlier_clip   = "percentile"

The only reason this exists (instead of just env vars on nnUNetTrainerNyul)
is that nnU-Net uses `self.__class__.__name__` for the output folder name.
Two Nyul experiments with different anchor types need different class names
so they write to different folders.
"""
from .nnUNetTrainerNyul import nnUNetTrainerNyul


class nnUNetTrainerNyulIdentity(nnUNetTrainerNyul):
    """Nyul + identity anchor + outlier clip (= our prior 'Path AB' config)."""

    @property
    def anchor_type(self) -> str:
        return "identity"

    @property
    def outlier_clip(self) -> str:
        return "percentile"
