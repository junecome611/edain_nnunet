"""Trainer subclasses for EDAIN+nnU-Net.

nnUNetv2_train can discover these by name as long as the module is importable
in the Python environment. We rely on PYTHONPATH including the project root.
"""
from .nnUNetTrainerEDAIN import nnUNetTrainerEDAIN

__all__ = ["nnUNetTrainerEDAIN"]
