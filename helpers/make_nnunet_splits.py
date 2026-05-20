#!/usr/bin/env python
"""Convert lipo_split.json to nnUNet v2 splits_final.json.

Writes to $nnUNet_preprocessed/Dataset500_Lipo/splits_final.json.
Must run AFTER nnUNetv2_plan_and_preprocess and BEFORE any nnUNetv2_train call,
otherwise nnUNet generates its own KFold split on first training run.
"""
import json
import os
from pathlib import Path

DATASET_NAME = "Dataset500_Lipo"
SPLIT_JSON = Path(__file__).parent / "lipo_split.json"


def main():
    nn_pre = os.environ.get("nnUNet_preprocessed")
    if nn_pre is None:
        raise RuntimeError("env var nnUNet_preprocessed is not set")
    dst = Path(nn_pre) / DATASET_NAME / "splits_final.json"
    if not dst.parent.exists():
        raise RuntimeError(
            f"{dst.parent} does not exist; run nnUNetv2_plan_and_preprocess first"
        )

    with open(SPLIT_JSON) as f:
        splits = json.load(f)
    all_subjects = set()
    for fold in splits.values():
        all_subjects.update(fold["val_subjects_sorted"])

    out = []
    for fold_idx in ["0", "1", "2", "3", "4"]:
        val = sorted(splits[fold_idx]["val_subjects_sorted"])
        train = sorted(all_subjects - set(val))
        out.append({"train": train, "val": val})
        print(f"[splits] fold {fold_idx}: train={len(train)} val={len(val)}")

    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[splits] wrote {dst}")


if __name__ == "__main__":
    main()
