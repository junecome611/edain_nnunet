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


def precompute_v1_stats_from_npz(
    preprocessed_dir: Path,
    train_case_ids: List[str],
    method: str = "otsu",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute per-case EDAIN v1 stats from nnU-Net's preprocessed cases.

    Works with BOTH nnU-Net storage formats:
        - nnU-Net <= 2.6 wrote .npz files (nnUNetDatasetNumpy)
        - nnU-Net >= 2.7 writes .b2nd files (nnUNetDatasetBlosc2, blosc2 backend)
    We delegate the format detection + loading to nnU-Net's `infer_dataset_class`
    so this stays compatible across versions.

    Args:
        preprocessed_dir:   directory containing per-case nnU-Net preprocessed
                            artifacts (.b2nd + .pkl OR .npz + .pkl).
        train_case_ids:     list like ['Lipo-001', 'Lipo-002', ...] matching
                            the keys in splits_final.json.
        method:             'otsu' or 'nonzero' foreground mask.
        verbose:            print progress.

    Returns:
        dict {case_id -> {'fg_mean', 'fg_std', 'fg_p2', 'fg_p98', ...}}
    """
    # Auto-detect b2nd vs npz storage. The returned class has a .load_case
    # method that yields (data, seg, seg_prev, properties).
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_cls = infer_dataset_class(str(preprocessed_dir))
    if verbose:
        print(f"[precompute] detected dataset class: {dataset_cls.__name__}")
    ds = dataset_cls(str(preprocessed_dir), identifiers=list(train_case_ids))

    stats: Dict[str, Dict[str, float]] = {}
    n_failed = 0
    n = len(train_case_ids)
    for i, cid in enumerate(train_case_ids):
        try:
            data, _seg, _prev, _props = ds.load_case(cid)
            # data shape: (C, D, H, W). blosc2 returns a lazy NDArray;
            # np.asarray materialises it to a real ndarray for percentile ops.
            volume = np.asarray(data[0])
            s = compute_per_case_stats(volume, method=method)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{n}] {cid}: FAILED {type(e).__name__}: {e}")
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
    # Include BOTH train and val cases. The trainer needs train stats during
    # training, and the same wrapper needs val stats during the post-training
    # `perform_actual_validation` sliding-window inference. If val stats are
    # missing, the wrapper silently falls back to a default (mean across
    # train cases) that can be severely distorted by intensity-outlier cases
    # — observed train-eval gap of ~0.12 dice on fold 0.
    train_ids = splits[args.fold]["train"]
    val_ids   = splits[args.fold]["val"]
    all_ids = list(train_ids) + list(val_ids)
    print(f"[precompute] fold {args.fold}: {len(train_ids)} train + "
          f"{len(val_ids)} val = {len(all_ids)} cases total")
    print(f"[precompute] reading from {args.preprocessed_dir}")

    if not args.preprocessed_dir.exists():
        raise FileNotFoundError(
            f"preprocessed_dir does not exist: {args.preprocessed_dir}\n"
            f"Did you run `nnUNetv2_preprocess -d <id> -c 3d_fullres "
            f"-plans_name nnUNetPlans_raw`?"
        )

    stats = precompute_v1_stats_from_npz(
        preprocessed_dir=args.preprocessed_dir,
        train_case_ids=all_ids,
        method=args.method,
    )
    if len(stats) == 0:
        # Sanity check: 0 cases means the preprocessed_dir didn't contain any
        # readable case files. The trainer would silently fall back to
        # default-stats and the network would NaN out within a few epochs.
        # Fail loudly instead so the user catches this at setup time.
        found_npz = sorted(args.preprocessed_dir.glob("*.npz"))
        found_b2nd = sorted(args.preprocessed_dir.glob("*.b2nd"))
        found_pkl = sorted(args.preprocessed_dir.glob("*.pkl"))
        raise RuntimeError(
            f"precompute produced 0 stats. Expected to find case files matching "
            f"the {len(all_ids)} case IDs from "
            f"{args.splits_json} (e.g. '{all_ids[0]}.b2nd' or '{all_ids[0]}.npz') "
            f"under {args.preprocessed_dir}.\n"
            f"What's actually there:\n"
            f"  {len(found_b2nd)} .b2nd file(s); first few: "
            f"{[p.name for p in found_b2nd[:5]]}\n"
            f"  {len(found_npz)} .npz file(s); first few: "
            f"{[p.name for p in found_npz[:5]]}\n"
            f"  {len(found_pkl)} .pkl file(s)\n"
            f"Most likely causes:\n"
            f"  1. file naming mismatch: splits_final.json has IDs like "
            f"'{all_ids[0]}' but the per-case files have different names "
            f"(e.g. with an extra '_MR_1' suffix).\n"
            f"  2. raw preprocessing wrote to a different directory than "
            f"expected. Check the plans JSON's `data_identifier`."
        )
    save_stats_dict(stats, args.output_json)
    n_train_ok = sum(1 for c in train_ids if c in stats)
    n_val_ok   = sum(1 for c in val_ids   if c in stats)
    print(f"[precompute] saved {len(stats)} cases ({n_train_ok}/{len(train_ids)} "
          f"train, {n_val_ok}/{len(val_ids)} val) -> {args.output_json}")


if __name__ == "__main__":
    main()
