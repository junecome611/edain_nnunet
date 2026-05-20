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
        self._current_case_ids = case_ids

    def _lookup_stats(self, batch_size: int, device) -> torch.Tensor:
        ids = self._current_case_ids
        if not ids or not self.case_stats_table:
            return self._default_stats.to(device).unsqueeze(0).expand(batch_size, -1)
        rows = []
        for cid in ids[:batch_size]:
            if cid in self.case_stats_table:
                rows.append(self.case_stats_table[cid])
            else:
                rows.append(self._default_stats.cpu())
        return torch.stack(rows, dim=0).to(device).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fg_stats = self._lookup_stats(x.shape[0], x.device)
        x_norm = self.edain(x, fg_stats)
        return self.backbone(x_norm)
