"""Precompute EDAIN artifacts on nnU-Net's preprocessed cases.

The crucial property here is that we read the EXACT same per-case tensor that
nnU-Net's training loop will see (after z-score, foreground crop, etc.).
This eliminates the train/inference gamma-domain mismatch (bug A1) that
plagued the old custom-MONAI pipeline.

Works with BOTH nnU-Net storage formats:
    - nnU-Net <= 2.6 wrote .npz files (nnUNetDatasetNumpy)
    - nnU-Net >= 2.7 writes .b2nd files (nnUNetDatasetBlosc2, blosc2 backend)
We delegate format detection + loading to nnU-Net's `infer_dataset_class`
so this stays compatible across versions.

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


def precompute_from_nnunet_preprocessed(
    preprocessed_dir: Path,
    train_case_ids: List[str],
    *,
    lookup_only_case_ids: Optional[List[str]] = None,
    anchor_type: str = "identity",
    outlier_clip: str = "none",
    K: int = 9,
    B_supp: float = 4.0,
    percentiles: Tuple[float, ...] = PERCENTILES,
    verbose: bool = True,
) -> dict:
    """Compute gamma per case; fit standardizer + theta_0 on train cases only.

    Args:
        preprocessed_dir:     e.g. .../nnUNet_preprocessed/Dataset500_Lipo/
                              nnUNetPlans_3d_fullres
        train_case_ids:       cases used for FITTING standardizer + theta_0.
                              Must be exclusively training cases — including
                              val cases here leaks them into the spline anchor.
        lookup_only_case_ids: extra cases (typically val) whose gamma is
                              computed and added to the lookup table but NOT
                              used for fitting. Needed so the wrapper has the
                              right gamma during sliding-window inference.
        anchor_type:          'identity' or 'population_nyul'
        outlier_clip:         'none' (data already passed through nnU-Net
                              ZScoreNormalization, no extra clip) or
                              'percentile' (re-clip foreground at 0.5/99.5
                              before computing gamma)
        K, B_supp:            spline grid
        verbose:              print progress

    Returns:
        dict with 'standardizer_mu', 'standardizer_sigma', 'theta_0', 'r_0',
        'population_landmarks', 'case_gammas' (dict[case_id -> tensor]).
    """
    lookup_only_case_ids = list(lookup_only_case_ids or [])
    all_case_ids = list(train_case_ids) + lookup_only_case_ids
    n_train = len(train_case_ids)
    if verbose:
        print(f"[precompute] reading {n_train} train + "
              f"{len(lookup_only_case_ids)} lookup-only cases from "
              f"{preprocessed_dir}")

    # Auto-detect b2nd vs npz storage. The returned class has a .load_case
    # method that yields (data, seg, seg_prev, properties).
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_cls = infer_dataset_class(str(preprocessed_dir))
    if verbose:
        print(f"[precompute] detected dataset class: {dataset_cls.__name__}")
    ds = dataset_cls(str(preprocessed_dir), identifiers=all_case_ids)

    case_gammas: Dict[str, torch.Tensor] = {}
    gamma_rows: List[torch.Tensor] = []  # train-only, for fitting
    n_failed = 0

    for i, cid in enumerate(all_case_ids):
        is_train = i < n_train
        try:
            data, _seg, _prev, _props = ds.load_case(cid)
            # data shape: (C, D, H, W). blosc2 returns a lazy NDArray;
            # np.asarray materialises it to a real ndarray.
            volume = np.asarray(data[0])
            X = torch.from_numpy(volume).float()
            mask = X != 0.0
            if outlier_clip == "percentile":
                fg = X[mask]
                # torch.quantile has a hard 2**24 element cap (silent OOM-ish
                # crash with "input tensor is too large" above that). For
                # large Lipo cases (>17M foreground voxels) we have to
                # subsample first. Random with-replacement sampling is
                # statistically tight for percentile estimation at this
                # sample size.
                _MAX_Q = (1 << 24) - 1
                fg_for_q = fg.float()
                if fg_for_q.numel() > _MAX_Q:
                    idx = torch.randint(0, fg_for_q.numel(), (_MAX_Q,),
                                        device=fg_for_q.device)
                    fg_for_q = fg_for_q[idx]
                lo, hi = torch.quantile(
                    fg_for_q,
                    torch.tensor([0.005, 0.995], dtype=torch.float32),
                )
                X = X.clamp(min=lo.item(), max=hi.item())
            gamma = percentile_summary(X, mask, percentiles=percentiles)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{i+1}/{len(all_case_ids)}] FAILED {cid}: "
                      f"{type(e).__name__}: {e}")
            continue

        case_gammas[cid] = gamma
        if is_train:
            gamma_rows.append(gamma)
        if verbose and (i == 0 or (i + 1) % max(1, len(all_case_ids) // 10) == 0):
            kind = "train" if is_train else "lookup"
            print(f"  [{i+1}/{len(all_case_ids)}] {cid} ({kind}): "
                  f"gamma[0,5,10]=({gamma[0]:.3f},{gamma[5]:.3f},{gamma[10]:.3f})")

    if len(gamma_rows) < 2:
        # Same defensive diagnostics as the v1 precompute: surface what's
        # actually on disk so the user can spot naming / data_identifier
        # mismatches at setup time instead of NaN-ing during training.
        found_npz = sorted(preprocessed_dir.glob("*.npz"))
        found_b2nd = sorted(preprocessed_dir.glob("*.b2nd"))
        found_pkl = sorted(preprocessed_dir.glob("*.pkl"))
        first_id = train_case_ids[0] if train_case_ids else "<none>"
        raise RuntimeError(
            f"only {len(gamma_rows)} cases ok; cannot fit. Expected per-case "
            f"files matching the {len(train_case_ids)} training IDs (e.g. "
            f"'{first_id}.b2nd' or '{first_id}.npz') under {preprocessed_dir}.\n"
            f"What's actually there:\n"
            f"  {len(found_b2nd)} .b2nd file(s); first few: "
            f"{[p.name for p in found_b2nd[:5]]}\n"
            f"  {len(found_npz)} .npz file(s); first few: "
            f"{[p.name for p in found_npz[:5]]}\n"
            f"  {len(found_pkl)} .pkl file(s)\n"
            f"Most likely causes:\n"
            f"  1. file naming mismatch between splits_final.json IDs and "
            f"per-case files.\n"
            f"  2. preprocessing wrote to a different directory than expected. "
            f"Check the plans JSON's `data_identifier`.\n"
            f"  3. {n_failed} case(s) failed to load — check the FAILED lines "
            f"above."
        )

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
