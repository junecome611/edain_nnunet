"""Precompute EDAIN artifacts on nnU-Net's preprocessed .npz files.

The crucial property here is that we read the EXACT same per-case tensor that
nnU-Net's training loop will see (after z-score, foreground crop, etc.).
This eliminates the train/inference gamma-domain mismatch (bug A1) that
plagued the old custom-MONAI pipeline.

Output:
    PrecomputeArtifacts with:
        standardizer_mu, standardizer_sigma  (11,)
        theta_0                              (3K - 1,)
        case_gammas                          dict[case_id -> (11,) tensor]
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from mri_edain_v2.modules.percentile import PERCENTILES, percentile_summary
from mri_edain_v2.modules.standardizer import CoordinateStandardizer
from mri_edain_v2.modules.nyul_init import (
    fit_population_nyul_theta_0,
    compute_non_affineness,
)


def _load_nnunet_case(npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a nnU-Net preprocessed case. Returns (X, mask).

    nnU-Net stores preprocessed data as 'data' of shape (C, D, H, W) and a
    'seg' channel. Background after z-score (with use_mask_for_norm) is 0;
    we use (X != 0) as the foreground mask, same as the trainer will see.
    """
    arr = np.load(str(npz_path))
    data = arr["data"]  # (C, D, H, W), already z-scored
    X = torch.from_numpy(data[0]).float()
    mask = X != 0.0
    return X, mask


def precompute_from_nnunet_preprocessed(
    preprocessed_dir: Path,
    train_case_ids: List[str],
    *,
    anchor_type: str = "identity",
    outlier_clip: str = "none",
    K: int = 9,
    B_supp: float = 4.0,
    percentiles: Tuple[float, ...] = PERCENTILES,
    verbose: bool = True,
) -> dict:
    """Compute gamma per training case, fit standardizer + theta_0.

    Args:
        preprocessed_dir: e.g. .../nnUNet_preprocessed/Dataset500_Lipo/
                          nnUNetPlans_3d_fullres
        train_case_ids:   list like ['Lipo-002_MR_1', ...] from splits_final.json
        anchor_type:      'identity' or 'population_nyul'
        outlier_clip:     'none' (data already passed through nnU-Net
                          ZScoreNormalization, no extra clip) or
                          'percentile' (re-clip foreground at 0.5/99.5
                          before computing gamma)
        K, B_supp:        spline grid
        verbose:          print progress

    Returns:
        dict with 'standardizer_mu', 'standardizer_sigma', 'theta_0', 'r_0',
        'population_landmarks', 'case_gammas' (dict[case_id -> tensor]).
    """
    if verbose:
        print(f"[precompute] reading {len(train_case_ids)} cases from "
              f"{preprocessed_dir}")

    case_gammas: Dict[str, torch.Tensor] = {}
    gamma_rows: List[torch.Tensor] = []
    n_failed = 0

    for i, cid in enumerate(train_case_ids):
        npz_path = preprocessed_dir / f"{cid}.npz"
        if not npz_path.exists():
            n_failed += 1
            if verbose:
                print(f"  [{i+1}/{len(train_case_ids)}] MISSING: {npz_path.name}")
            continue
        try:
            X, mask = _load_nnunet_case(npz_path)
            if outlier_clip == "percentile":
                fg = X[mask]
                lo, hi = torch.quantile(
                    fg.float(),
                    torch.tensor([0.005, 0.995], dtype=torch.float32),
                )
                X = X.clamp(min=lo.item(), max=hi.item())
            gamma = percentile_summary(X, mask, percentiles=percentiles)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{i+1}/{len(train_case_ids)}] FAILED {cid}: {e}")
            continue

        case_gammas[cid] = gamma
        gamma_rows.append(gamma)
        if verbose and (i == 0 or (i + 1) % max(1, len(train_case_ids) // 10) == 0):
            print(f"  [{i+1}/{len(train_case_ids)}] {cid}: "
                  f"gamma[0,5,10]=({gamma[0]:.3f},{gamma[5]:.3f},{gamma[10]:.3f})")

    if len(gamma_rows) < 2:
        raise RuntimeError(f"only {len(gamma_rows)} cases ok; cannot fit")

    raw_gammas = torch.stack(gamma_rows, dim=0)

    # Fit per-coordinate standardizer.
    standardizer = CoordinateStandardizer(n_dim=len(percentiles))
    standardizer.fit(raw_gammas)
    if verbose:
        print(f"[precompute] standardizer fit on {raw_gammas.shape[0]} cases")

    # Fit theta_0.
    if verbose:
        print(f"[precompute] fitting theta_0 (anchor_type={anchor_type})")
    theta_0 = fit_population_nyul_theta_0(
        raw_gammas, K=K, B_supp=B_supp,
        anchor_type=anchor_type, verbose=verbose,
    )
    r_0 = compute_non_affineness(theta_0, K=K, B_supp=B_supp).item()
    if verbose:
        print(f"[precompute] theta_0 r_0 = {r_0:.4f}")

    return {
        "standardizer_mu": standardizer.mu.detach().clone(),
        "standardizer_sigma": standardizer.sigma.detach().clone(),
        "theta_0": theta_0.detach().clone(),
        "r_0": r_0,
        "population_landmarks": raw_gammas.mean(dim=0).detach().clone(),
        "case_gammas": case_gammas,
        "K": K,
        "B_supp": B_supp,
        "anchor_type": anchor_type,
        "outlier_clip": outlier_clip,
        "n_failed": n_failed,
    }


def save_artifacts(artifacts: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, str(path))


def load_artifacts(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)
