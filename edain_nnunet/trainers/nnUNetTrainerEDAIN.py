"""nnUNetTrainerEDAIN: nnU-Net + MRIEDAIN v2 layer.

This trainer inherits 99% of its behaviour from nnU-Net's nnUNetTrainer.
The ONLY differences are:

    1. `initialize()` is extended to:
       a) read the per-fold training cases
       b) precompute EDAIN artifacts (standardizer mu/sigma, theta_0,
          per-case whole-volume gamma) on nnU-Net's preprocessed .npz files
       c) build a MRIEDAINLayer and wrap self.network with EDAINWrapper

    2. `train_step()` and `validation_step()` push the current batch's
       case_ids to the wrapper so it can look up whole-volume gammas
       (fix for bug A1).

Everything else -- data augmentation, optimizer (SGD nesterov 0.99,
weight_decay 3e-5), poly LR, DC+CE loss, deep supervision, mirroring at
inference, EMA, AMP -- is exactly the same as the upstream nnU-Net
baseline.

Configuration: via environment variables read in initialize():
    EDAIN_ANCHOR_TYPE   = 'identity' | 'population_nyul'   (default: identity)
    EDAIN_OUTLIER_CLIP  = 'none' | 'percentile'            (default: none)
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# EDAIN modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mri_edain_v2.modules.standardizer import CoordinateStandardizer
from mri_edain_v2.modules.edain_layer import MRIEDAINLayer

from edain_nnunet.network.edain_wrapper import EDAINWrapper
from edain_nnunet.precompute.nnunet_precompute import (
    precompute_from_nnunet_preprocessed,
    save_artifacts,
    load_artifacts,
)


class nnUNetTrainerEDAIN(nnUNetTrainer):
    """nnU-Net + EDAIN v2 layer (input-conditional monotonic spline)."""

    # ----- knobs read from env (so we don't need a custom CLI) -----------
    @property
    def anchor_type(self) -> str:
        return os.environ.get("EDAIN_ANCHOR_TYPE", "identity")

    @property
    def outlier_clip(self) -> str:
        return os.environ.get("EDAIN_OUTLIER_CLIP", "none")

    # ----- override: initialize ------------------------------------------
    def initialize(self):
        # First do everything nnU-Net normally does: build network, loss,
        # optimizer, etc. After this call self.network is a plain nnU-Net.
        super().initialize()

        # Now precompute EDAIN artifacts, then wrap self.network.
        artifact_path = (
            Path(self.output_folder).parent / "edain_artifacts" /
            f"fold_{self.fold}_{self.anchor_type}_{self.outlier_clip}.pt"
        )
        if artifact_path.exists():
            self.print_to_log_file(f"[EDAIN] loading cached artifacts: {artifact_path}")
            artifacts = load_artifacts(artifact_path)
        else:
            artifacts = self._precompute_edain()
            save_artifacts(artifacts, artifact_path)
            self.print_to_log_file(f"[EDAIN] saved artifacts: {artifact_path}")

        # Build the EDAIN layer with fitted buffers.
        standardizer = CoordinateStandardizer(
            n_dim=artifacts["standardizer_mu"].shape[0]
        )
        standardizer.mu.copy_(artifacts["standardizer_mu"])
        standardizer.sigma.copy_(artifacts["standardizer_sigma"])
        standardizer.is_fit.fill_(1)

        edain = MRIEDAINLayer(
            standardizer=standardizer,
            theta_0=artifacts["theta_0"],
            K=artifacts["K"],
            B_supp=artifacts["B_supp"],
        )

        # Wrap the (already built) backbone with EDAIN.
        self.network = EDAINWrapper(
            edain=edain,
            backbone=self.network,
            case_gamma_table=artifacts["case_gammas"],
        ).to(self.device)

        # Recreate optimizer because self.network changed.
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.print_to_log_file(
            f"[EDAIN] anchor={self.anchor_type} clip={self.outlier_clip} "
            f"r_0={artifacts['r_0']:.4f} "
            f"n_cases={len(artifacts['case_gammas'])}"
        )

    # ----- precompute helper --------------------------------------------
    def _precompute_edain(self) -> dict:
        """Read splits_final.json, locate preprocessed .npz, precompute."""
        nn_pre = os.environ["nnUNet_preprocessed"]
        ds_name = self.plans_manager.dataset_name
        splits_path = Path(nn_pre) / ds_name / "splits_final.json"
        with open(splits_path) as f:
            splits = json.load(f)
        train_ids = splits[self.fold]["train"]

        # The .npz path: nnUNet stores them as
        #     <ds_name>/<plans_identifier>_<configuration>/<case>.npz
        plans_id = self.plans_manager.plans_name
        config_name = self.configuration_name
        pre_dir = Path(nn_pre) / ds_name / f"{plans_id}_{config_name}"
        if not pre_dir.exists():
            # nnU-Net new layout: <ds_name>/<plans>_<config>/, but old runs
            # used <ds_name>/nnUNetPlans_<config>/. Try both.
            alt = Path(nn_pre) / ds_name / f"nnUNetPlans_{config_name}"
            if alt.exists():
                pre_dir = alt
            else:
                raise FileNotFoundError(
                    f"Could not find preprocessed dir; tried {pre_dir} and {alt}"
                )

        return precompute_from_nnunet_preprocessed(
            preprocessed_dir=pre_dir,
            train_case_ids=train_ids,
            anchor_type=self.anchor_type,
            outlier_clip=self.outlier_clip,
        )

    # ----- override: train_step / validation_step ------------------------
    # The only addition vs nnUNetTrainer.train_step is set_current_batch(...)
    # right before the forward pass, so the wrapper can lookup gammas.
    def train_step(self, batch: dict) -> dict:
        case_ids = batch.get("keys", None)
        if case_ids is not None and hasattr(self.network, "set_current_batch"):
            self.network.set_current_batch(case_ids)
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        case_ids = batch.get("keys", None)
        if case_ids is not None and hasattr(self.network, "set_current_batch"):
            self.network.set_current_batch(case_ids)
        return super().validation_step(batch)
