"""nnU-Net trainer subclasses for each experiment.

Trainer naming convention (matches the user's terminology):
    nnUNetTrainerEDAINv1        - classical 4-sublayer EDAIN (Sanna Passino 2024)
                                   h1 (outlier) + h2 (shift) + h3 (scale)
                                   NO power transform.
    nnUNetTrainerEDAINv1Power   - EDAIN v1 with all 4 sublayers (h1+h2+h3+h4)
                                   adds h4 (Yeo-Johnson power transform).
    nnUNetTrainerNyul           - base class for Nyul-style spline + hypernet.
                                   anchor_type configured via env var.
    nnUNetTrainerNyulIdentity   - Nyul with identity anchor + percentile clip
                                   (= our prior best 'Path AB').
    nnUNetTrainerNyulPopnyul    - Nyul with population_nyul anchor + no clip.

The Identity / Popnyul subclasses exist so nnU-Net's output_folder uses
different class names for the two Nyul experiments (output folders must differ).

For the vanilla nnU-Net baseline use upstream nnUNetTrainer (no subclass).
"""
from .nnUNetTrainerEDAINv1 import nnUNetTrainerEDAINv1
from .nnUNetTrainerEDAINv1Power import nnUNetTrainerEDAINv1Power
from .nnUNetTrainerNyul import nnUNetTrainerNyul
from .nnUNetTrainerNyulIdentity import nnUNetTrainerNyulIdentity
from .nnUNetTrainerNyulPopnyul import nnUNetTrainerNyulPopnyul

__all__ = [
    "nnUNetTrainerEDAINv1",
    "nnUNetTrainerEDAINv1Power",
    "nnUNetTrainerNyul",
    "nnUNetTrainerNyulIdentity",
    "nnUNetTrainerNyulPopnyul",
]
