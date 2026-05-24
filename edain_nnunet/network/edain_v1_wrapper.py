"""Network wrapper for EDAIN v1 + nnU-Net backbone.

Identical pattern to the Nyul wrapper, except the EDAIN v1 layer needs
per-image (fg_mean, fg_std, fg_p2, fg_p98) instead of (gamma_raw).
These stats are looked up by case_id from a precomputed table.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from mri_edain_v1.modules.edain_v1_layer import EDAINv1Layer


class EDAINv1Wrapper(nn.Module):
    """EDAIN-v1(X) -> backbone(X_normalized)."""

    def __init__(
        self,
        edain: EDAINv1Layer,
        backbone: nn.Module,
        case_stats_table: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.edain = edain
        self.backbone = backbone
        # case_stats_table[case_id] = tensor[4] in order (fg_mean, fg_std, fg_p2, fg_p98)
        self.case_stats_table = case_stats_table or {}
        self._current_case_ids = None
        # Default stats: used when case_id is missing (e.g., test-time-prediction).
        # If the table has any entries, use the mean of all known stats.
        if self.case_stats_table:
            stacked = torch.stack(list(self.case_stats_table.values()), dim=0)
            self.register_buffer("_default_stats", stacked.mean(dim=0))
        else:
            self.register_buffer(
                "_default_stats", torch.tensor([0.0, 1.0, 0.0, 1.0]))

    def set_current_batch(self, case_ids=None) -> None:
        # case_ids may be a python list (our tests) OR a numpy array (nnU-Net
        # data loader at runtime — see data_loader.py:207). Both are accepted.
        self._current_case_ids = case_ids

    def _lookup_stats(self, batch_size: int, device) -> torch.Tensor:
        ids = self._current_case_ids
        default_row = self._default_stats.cpu()
        # Handle ALL of: None / empty list / empty numpy array / empty table.
        # Note: `not numpy_array` raises ValueError for multi-element arrays,
        # so we explicitly use `len(...) == 0` after a None check.
        if ids is None or len(ids) == 0 or len(self.case_stats_table) == 0:
            # IMPORTANT: build a fresh contiguous tensor (stack repeats),
            # NOT an expand() with stride 0. Otherwise the downstream EDAIN
            # layer can hit "more than one element refers to single memory
            # location" on its in-place ops.
            rows = [default_row for _ in range(batch_size)]
            return torch.stack(rows, dim=0).to(device).float().contiguous()
        n_ids = len(ids)
        rows = []
        for i in range(batch_size):
            # Cycle through ids if batch_size > n_ids (e.g., sliding-window
            # inference reuses a single case's stats for several windows).
            cid = str(ids[i % n_ids])   # numpy.str_ -> str so dict lookup hits
            rows.append(self.case_stats_table.get(cid, default_row))
        return torch.stack(rows, dim=0).to(device).float().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fg_stats = self._lookup_stats(x.shape[0], x.device)
        x_norm = self.edain(x, fg_stats)
        return self.backbone(x_norm)

    # ------------------------------------------------------------------ #
    # Transparent attribute forwarding to backbone.
    #
    # Why this is needed:
    #   nnU-Net's nnUNetTrainer.set_deep_supervision_enabled() accesses
    #       self.network.decoder.deep_supervision = enabled
    #   on the assumption that self.network IS a PlainConvUNet. When we
    #   wrap it, `wrapper.decoder` would normally raise AttributeError,
    #   blowing up on_train_start.
    #
    # The fix: forward unknown attribute lookups to self.backbone. This
    # covers `.decoder`, `.encoder`, and anything else nnU-Net might add
    # in future minor versions, without us having to chase each one.
    #
    # We're careful not to recurse during __init__ (before `backbone` is
    # registered as a submodule), so we look it up via the underlying
    # _modules dict, never through __getattr__.
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "_modules")
            backbone = modules.get("backbone", None) if modules else None
            if backbone is None:
                raise AttributeError(
                    f"'{type(self).__name__}' has no attribute '{name}'"
                )
            return getattr(backbone, name)
