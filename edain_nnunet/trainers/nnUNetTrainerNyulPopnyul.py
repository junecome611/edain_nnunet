"""nnUNetTrainerNyulPopnyul — Nyul spline trainer with population_nyul anchor.

Thin subclass of nnUNetTrainerNyul. Sets:
    anchor_type    = "population_nyul"
    outlier_clip   = "none"

See nnUNetTrainerNyulIdentity.py for why subclassing (instead of env vars) is
needed: it keeps the output folder name distinct.
"""
from .nnUNetTrainerNyul import nnUNetTrainerNyul


class nnUNetTrainerNyulPopnyul(nnUNetTrainerNyul):
    """Nyul + classical population_nyul anchor + no outlier clip."""

    @property
    def anchor_type(self) -> str:
        return "population_nyul"

    @property
    def outlier_clip(self) -> str:
        return "none"
