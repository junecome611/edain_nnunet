"""Precompute per-case stats for EDAIN v1 from nnU-Net's RAW preprocessed .npz.

This script must be run AFTER `nnUNetv2_plan_and_preprocess -p nnUNetPlans_raw`,
which generates raw (no z-score) preprocessed .npz files. We then iterate over
the training cases, compute Otsu-foreground statistics, and cache them in a
JSON file that the trainer loads at startup.

USAGE
=====
    python -m mri_edain_v1.precompute.precompute_v1_stats \
        --preprocessed_dir $nnUNet_preprocessed/Dataset500_Lipo/nnUNetPlans_raw_3d_fullres \
        --splits_json     $nnUNet_preprocessed/Dataset500_Lipo/splits_final.json \
        --output_json     ./edain_v1_stats_fold0.json \
        --fold 0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from mri_edain_v1.modules.percentile_stats import (
    compute_per_case_stats, save_stats_dict
)


def _load_case_data(npz_path: Path) -> np.ndarray:
    """Load nnU-Net preprocessed .npz file. Returns the image data (C, D, H, W)."""
    arr = np.load(str(npz_path))
    return arr["data"]


def precompute_v1_stats_from_npz(
    preprocessed_dir: Path,
    train_case_ids: List[str],
    method: str = "otsu",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute per-case EDAIN v1 stats over a list of case IDs.

    Args:
        preprocessed_dir:   directory containing <case_id>.npz files
        train_case_ids:     list like ['Lipo-002_MR_1', ...]
        method:             'otsu' or 'nonzero'
        verbose:            print progress

    Returns:
        dict {case_id -> {'fg_mean', 'fg_std', 'fg_p2', 'fg_p98', ...}}
    """
    stats: Dict[str, Dict[str, float]] = {}
    n_failed = 0
    n = len(train_case_ids)
    for i, cid in enumerate(train_case_ids):
        npz_path = preprocessed_dir / f"{cid}.npz"
        if not npz_path.exists():
            if verbose:
                print(f"  [{i+1}/{n}] {cid}: MISSING {npz_path.name}")
            n_failed += 1
            continue
        try:
            data = _load_case_data(npz_path)
            # nnU-Net stores (C, D, H, W); EDAIN v1 is single-channel
            volume = data[0]
            s = compute_per_case_stats(volume, method=method)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{n}] {cid}: FAILED {e}")
            n_failed += 1
            continue
        stats[cid] = s
        if verbose and (i == 0 or (i + 1) % max(1, n // 10) == 0 or i == n - 1):
            print(f"  [{i+1}/{n}] {cid}: "
                  f"fg_mean={s['fg_mean']:.1f} fg_std={s['fg_std']:.1f} "
                  f"p2={s['fg_p2']:.1f} p98={s['fg_p98']:.1f}")
    if verbose:
        print(f"[precompute] done: {len(stats)} ok, {n_failed} failed")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed_dir", type=Path, required=True,
                    help="Path to .../Dataset500_Lipo/nnUNetPlans_raw_3d_fullres")
    ap.add_argument("--splits_json", type=Path, required=True,
                    help="Path to .../Dataset500_Lipo/splits_final.json")
    ap.add_argument("--output_json", type=Path, required=True,
                    help="Where to write the {case_id: stats} JSON")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--method", type=str, default="otsu",
                    choices=["otsu", "nonzero"])
    args = ap.parse_args()

    with open(args.splits_json) as f:
        splits = json.load(f)
    if not 0 <= args.fold < len(splits):
        raise ValueError(f"fold {args.fold} out of range (have {len(splits)} splits)")
    train_ids = splits[args.fold]["train"]
    print(f"[precompute] fold {args.fold}: {len(train_ids)} training cases")
    print(f"[precompute] reading from {args.preprocessed_dir}")

    stats = precompute_v1_stats_from_npz(
        preprocessed_dir=args.preprocessed_dir,
        train_case_ids=train_ids,
        method=args.method,
    )
    save_stats_dict(stats, args.output_json)
    print(f"[precompute] saved -> {args.output_json}")


if __name__ == "__main__":
    main()
