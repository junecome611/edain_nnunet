"""EDAIN network wrapper for nnU-Net integration.

Wraps a fitted MRIEDAINLayer in front of any nnU-Net backbone. The only
non-trivial bit is the optional per-case whole-volume gamma lookup, which is
the fix for bug A1 (train uses patch-gamma, val uses whole-volume gamma).

This file is single-purpose; everything else is inherited from nnU-Net.
"""
from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn

# EDAIN lives in the project-level mri_edain_v2 package. We import lazily to
# avoid any chance of altering its namespace.
from mri_edain_v2.modules.edain_layer import MRIEDAINLayer


class EDAINWrapper(nn.Module):
    """EDAIN(X) -> backbone(X_tilde).

    Args:
        edain:    fitted MRIEDAINLayer (theta_0 and standardizer buffers
                  already loaded from precompute artifact).
        backbone: the nnU-Net network returned by build_network_architecture.
        case_gamma_table:
                  optional dict {case_id -> (11,) tensor} of whole-volume
                  gamma_raw. If provided AND the batch has 'case_ids', train
                  will use these instead of computing gamma on the patch
                  (fix for bug A1). Validation always uses these.
    """

    def __init__(
        self,
        edain: MRIEDAINLayer,
        backbone: nn.Module,
        case_gamma_table: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.edain = edain
        self.backbone = backbone
        self.case_gamma_table = case_gamma_table or {}

    # ------------------------------------------------------------------ #
    # nnU-Net's train_step / validation_step call self.network(data) with
    # data shape (B, C, D, H, W) and no extra kwargs. To pass case_ids we
    # stash them on the wrapper via set_current_batch() right before the
    # forward call (done from a thin trainer hook).
    # ------------------------------------------------------------------ #

    def set_current_batch(self, case_ids: Optional[list] = None) -> None:
        self._current_case_ids = case_ids

    def _lookup_gammas(self, batch_size: int, device) -> Optional[torch.Tensor]:
        ids = getattr(self, "_current_case_ids", None)
        if not ids or not self.case_gamma_table:
            return None
        try:
            rows = [self.case_gamma_table[i] for i in ids[:batch_size]]
        except KeyError:
            return None
        return torch.stack(rows, dim=0).to(device).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0.0
        gamma_raw = self._lookup_gammas(x.shape[0], x.device)
        x_tilde, _ = self.edain(
            x, mask=mask, gamma_raw=gamma_raw, return_diagnostics=False
        )
        return self.backbone(x_tilde)

    # ------------------------------------------------------------------ #
    # Forward unknown attribute lookups to backbone. Specifically needed
    # because nnU-Net's set_deep_supervision_enabled accesses
    # `self.network.decoder.deep_supervision`. See EDAINv1Wrapper for
    # the full explanation.
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
