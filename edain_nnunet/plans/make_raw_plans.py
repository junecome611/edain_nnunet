"""Generate a no-normalization variant of nnU-Net's default plans.

This is the bridge between "nnU-Net wants to z-score by default" and
"our EDAIN layer wants to see RAW intensities". We make a copy of
nnUNetPlans.json, change `normalization_schemes` to `NoNormalization`,
and save under a new plans name. nnU-Net then produces a second set
of preprocessed .npz files (raw, not z-scored) under
    $nnUNet_preprocessed/<dataset>/<plans_name>_<configuration>/

USAGE
=====
    # After running:
    #   nnUNetv2_plan_and_preprocess -d 500 -c 3d_fullres
    # then:
    python -m edain_nnunet.plans.make_raw_plans -d 500 \\
        --src_plans_name nnUNetPlans \\
        --dst_plans_name nnUNetPlans_raw
    # then:
    nnUNetv2_preprocess -d 500 -c 3d_fullres -plans_name nnUNetPlans_raw

This produces:
    $nnUNet_preprocessed/Dataset500_Lipo/
        ├── nnUNetPlans.json                          (z-scored)
        ├── nnUNetPlans_3d_fullres/                   (z-scored .npz)
        ├── nnUNetPlans_raw.json                      (raw)        ← created here
        └── nnUNetPlans_raw_3d_fullres/               (raw .npz)   ← created by preprocess
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("--src_plans_name", type=str, default="nnUNetPlans")
    ap.add_argument("--dst_plans_name", type=str, default="nnUNetPlans_raw")
    ap.add_argument("--configurations", nargs="+", default=["3d_fullres"],
                    help="which configurations to swap normalization on")
    ap.add_argument("--ds_name", type=str, default=None,
                    help="Dataset name (auto-detect if omitted)")
    args = ap.parse_args()

    nn_pre = os.environ.get("nnUNet_preprocessed")
    if nn_pre is None:
        raise RuntimeError("env var nnUNet_preprocessed is not set")
    nn_pre = Path(nn_pre)

    # Auto-detect dataset name if not given
    if args.ds_name is None:
        candidates = sorted(nn_pre.glob(f"Dataset{args.dataset_id:03d}_*"))
        if not candidates:
            raise RuntimeError(
                f"No Dataset{args.dataset_id:03d}_* in {nn_pre}. "
                f"Did you run nnUNetv2_plan_and_preprocess?"
            )
        ds_dir = candidates[0]
    else:
        ds_dir = nn_pre / args.ds_name
    print(f"[plans] dataset dir = {ds_dir}")

    src_plans_path = ds_dir / f"{args.src_plans_name}.json"
    dst_plans_path = ds_dir / f"{args.dst_plans_name}.json"
    if not src_plans_path.exists():
        raise FileNotFoundError(
            f"{src_plans_path} not found. Run `nnUNetv2_plan_and_preprocess` first."
        )

    with open(src_plans_path) as f:
        plans = json.load(f)

    # Swap normalization for the chosen configurations
    for cfg in args.configurations:
        if cfg not in plans["configurations"]:
            print(f"[plans] WARNING: configuration '{cfg}' not in plans, skipping")
            continue
        schemes = plans["configurations"][cfg].get("normalization_schemes", [])
        new_schemes = ["NoNormalization"] * len(schemes)
        plans["configurations"][cfg]["normalization_schemes"] = new_schemes
        print(f"[plans] {cfg}: normalization {schemes} -> {new_schemes}")
        # use_mask_for_norm: set to False for NoNormalization channels
        use_mask = plans["configurations"][cfg].get("use_mask_for_norm", [])
        if use_mask:
            plans["configurations"][cfg]["use_mask_for_norm"] = [False] * len(use_mask)

    plans["plans_name"] = args.dst_plans_name

    with open(dst_plans_path, "w") as f:
        json.dump(plans, f, indent=2)
    print(f"[plans] saved -> {dst_plans_path}")
    print()
    print("Next step: run")
    print(f"  nnUNetv2_preprocess -d {args.dataset_id} -c {' '.join(args.configurations)} "
          f"-plans_name {args.dst_plans_name}")


if __name__ == "__main__":
    main()
