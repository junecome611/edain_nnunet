#!/usr/bin/env python
"""Create nnUNet v2 raw dataset layout for Lipo via symlinks.

Source:  /trinity/home/r112643/dataset/lipo/Lipo-XXX_MR_1_{image,segmentation}.nii.gz
Target:  $nnUNet_raw/Dataset500_Lipo/{imagesTr,labelsTr}/Lipo-XXX[_0000].nii.gz

Subjects are taken from lipo_split.json (Lipo-073 already excluded there).
Idempotent: re-running replaces existing symlinks.
"""
import json
import os
from pathlib import Path

SRC = Path("/trinity/home/r112643/dataset/lipo")
DATASET_NAME = "Dataset500_Lipo"
SPLIT_JSON = Path(__file__).parent / "lipo_split.json"


def main():
    nn_raw = os.environ.get("nnUNet_raw")
    if nn_raw is None:
        raise RuntimeError("env var nnUNet_raw is not set")
    dst_root = Path(nn_raw) / DATASET_NAME
    imgs_dir = dst_root / "imagesTr"
    lbls_dir = dst_root / "labelsTr"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_JSON) as f:
        splits = json.load(f)
    subjects = set()
    for fold in splits.values():
        subjects.update(fold["val_subjects_sorted"])
    subjects = sorted(subjects)
    print(f"[prepare] {len(subjects)} subjects from {SPLIT_JSON.name}")

    missing = []
    for sid in subjects:
        img_src = SRC / f"{sid}_MR_1_image.nii.gz"
        lbl_src = SRC / f"{sid}_MR_1_segmentation.nii.gz"
        if not img_src.exists() or not lbl_src.exists():
            missing.append(sid)
            continue
        img_dst = imgs_dir / f"{sid}_0000.nii.gz"
        lbl_dst = lbls_dir / f"{sid}.nii.gz"
        for d in (img_dst, lbl_dst):
            if d.is_symlink() or d.exists():
                d.unlink()
        img_dst.symlink_to(img_src.resolve())
        lbl_dst.symlink_to(lbl_src.resolve())
    if missing:
        raise RuntimeError(f"missing source files for: {missing}")
    print(f"[prepare] linked {len(subjects)} image+label pairs into {dst_root}")

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(subjects),
        "file_ending": ".nii.gz",
    }
    with open(dst_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"[prepare] wrote {dst_root / 'dataset.json'}")


if __name__ == "__main__":
    main()
