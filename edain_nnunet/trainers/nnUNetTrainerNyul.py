"""nnUNetTrainerNyul: nnU-Net + Nyúl-inspired learnable spline normalization.

This is what we previously called "MRI-EDAIN v2" — to avoid term confusion with
the original 4-sublayer EDAIN paper (which we now have a separate trainer for,
nnUNetTrainerEDAINv1), we rename this to "Nyul".

The Nyul layer = input-conditional monotonic RQ-spline:
    - 11 foreground percentiles -> hypernet -> Delta-theta (26 dims)
    - theta = theta_0 (frozen anchor) + Delta-theta
    - rational-quadratic spline parameterized by theta is applied voxel-wise
    - anchor choice: 'population_nyul' (paper Nyul) or 'identity' (no-op start)

Configuration via env vars:
    EDAIN_ANCHOR_TYPE   = 'identity' | 'population_nyul'   (default: identity)
    EDAIN_OUTLIER_CLIP  = 'none' | 'percentile'            (default: none)

REQUIREMENTS
============
Use the same z-scored preprocessing as vanilla nnU-Net (default plans).
Nyul reads gamma_raw from already-z-scored data.
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


class nnUNetTrainerNyul(nnUNetTrainer):
    """nnU-Net + Nyul-inspired input-conditional monotonic spline layer."""

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
        # Cache filename includes a schema-version suffix ('v2') because the
        # artifact dict layout changed when we added val-case gammas to the
        # lookup table. Old v1 caches would silently load with only train
        # cases and reproduce the inference bug.
        artifact_path = (
            Path(self.output_folder).parent / "edain_artifacts" /
            f"fold_{self.fold}_{self.anchor_type}_{self.outlier_clip}_v2.pt"
        )
        if artifact_path.exists():
            self.print_to_log_file(f"[Nyul] loading cached artifacts: {artifact_path}")
            artifacts = load_artifacts(artifact_path)
        else:
            artifacts = self._precompute_edain()
            save_artifacts(artifacts, artifact_path)
            self.print_to_log_file(f"[Nyul] saved artifacts: {artifact_path}")

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
            f"[Nyul] anchor={self.anchor_type} clip={self.outlier_clip} "
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
        # Val cases must also be in the gamma lookup table so that the wrapper
        # serves the correct per-case gamma during sliding-window inference.
        # Without this, every val case fell back to per-patch gamma (or worse,
        # to the wrong cached id), causing a ~0.12 train-eval dice gap on fold 0.
        val_ids = splits[self.fold]["val"]

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
            lookup_only_case_ids=val_ids,
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

    # ----- override: plumb case_id to wrapper during sliding-window inference -
    #
    # See nnUNetTrainerEDAINv1.perform_actual_validation for the full
    # rationale. Short version: nnU-Net's post-training validation loop calls
    # predictor.predict_sliding_window_return_logits(data) without ever
    # plumbing case_id to self.network. Our wrapper's _current_case_ids
    # stays stale, _lookup_gammas falls back to defaults, and every val case
    # gets the wrong gamma. Fix: swap dataset_class with a subclass whose
    # load_case() calls set_current_batch([k]) before yielding the data.
    def perform_actual_validation(self, save_probabilities: bool = False):
        original_cls = self.dataset_class
        network = self.network
        if not hasattr(network, "set_current_batch"):
            return super().perform_actual_validation(save_probabilities)

        class _CaseIDInjectingDataset(original_cls):
            def load_case(self_, k):
                network.set_current_batch([k])
                return super().load_case(k)

        self.dataset_class = _CaseIDInjectingDataset
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self.dataset_class = original_cls
