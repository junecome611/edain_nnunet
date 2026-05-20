"""Per-case foreground statistics for EDAIN v1.

EDAIN v1 needs per-image statistics on the RAW intensity values:
    - fg_mean:  mean over Otsu-foreground voxels   (used as mu_hat in h1)
    - fg_std:   std over Otsu-foreground voxels    (used by h3)
    - fg_p2:    2nd percentile of foreground       (used by EDAIN min-max wrapper)
    - fg_p98:   98th percentile of foreground      (used by EDAIN min-max wrapper)

These are computed ONCE per training case during precompute, then cached and
looked up by case_id at training/inference time.

Why precompute (not per-patch)?
    1. Patch is a small spatial crop -> percentile estimates noisy
    2. Patch percentiles AFTER augmentation are even further perturbed
    3. Validation uses whole-volume sliding window — match training domain
       (this is the fix for bug A1 from prior runs)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import torch

try:
    from skimage.filters import threshold_otsu
    _HAS_OTSU = True
except ImportError:
    _HAS_OTSU = False


def _compute_foreground_mask(volume: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Compute foreground mask. Returns boolean ndarray same shape as volume."""
    if method == "nonzero":
        return volume > 0
    if method == "otsu":
        if not _HAS_OTSU:
            raise RuntimeError(
                "skimage not installed; cannot use Otsu. Use method='nonzero' instead."
            )
        try:
            thr = threshold_otsu(volume)
        except Exception:
            thr = float(volume.mean())
        return volume > thr
    raise ValueError(f"unknown foreground method: {method}")


def compute_per_case_stats(
    volume: np.ndarray,
    method: str = "otsu",
    eps_std: float = 1e-6,
) -> Dict[str, float]:
    """Compute the 4 EDAIN statistics on one raw volume.

    Args:
        volume:  numpy array, shape (D, H, W) or (C, D, H, W) (single channel).
        method:  'otsu' (default) or 'nonzero'
        eps_std: clamp on the returned fg_std to avoid divide-by-zero downstream

    Returns:
        dict with keys: fg_mean, fg_std, fg_p2, fg_p98, fg_min, fg_max, n_fg
    """
    if volume.ndim == 4:
        if volume.shape[0] != 1:
            raise ValueError(f"expected single-channel volume, got shape {volume.shape}")
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"expected (D, H, W), got shape {volume.shape}")
    volume = volume.astype(np.float32)

    fg_mask = _compute_foreground_mask(volume, method=method)
    fg = volume[fg_mask]
    if fg.size < 100:
        # Degenerate case: fall back to whole-volume stats
        fg = volume.reshape(-1)

    fg_mean = float(fg.mean())
    fg_std = float(max(fg.std(), eps_std))
    fg_p2 = float(np.percentile(fg, 2.0))
    fg_p98 = float(np.percentile(fg, 98.0))
    fg_min = float(fg.min())
    fg_max = float(fg.max())

    return {
        "fg_mean": fg_mean,
        "fg_std": fg_std,
        "fg_p2": fg_p2,
        "fg_p98": fg_p98,
        "fg_min": fg_min,
        "fg_max": fg_max,
        "n_fg": int(fg.size),
    }


def save_stats_dict(stats: Dict[str, Dict[str, float]], path: Path) -> None:
    """Save {case_id -> stats} to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats_dict(path: Path) -> Dict[str, Dict[str, float]]:
    """Load {case_id -> stats} from JSON."""
    with open(path) as f:
        return json.load(f)


def stats_dict_to_tensor_lookup(
    stats: Dict[str, Dict[str, float]],
    keys: Tuple[str, ...] = ("fg_mean", "fg_std", "fg_p2", "fg_p98"),
) -> Dict[str, torch.Tensor]:
    """Convert {case_id -> {stat_name -> value}} into {case_id -> tensor[len(keys)]}.

    The tensor ordering matches `keys`. Default order: (fg_mean, fg_std, fg_p2, fg_p98).
    """
    out: Dict[str, torch.Tensor] = {}
    for cid, s in stats.items():
        out[cid] = torch.tensor([s[k] for k in keys], dtype=torch.float32)
    return out
