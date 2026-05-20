"""nnUNetTrainerEDAINv1 — nnU-Net + classical 4-sublayer EDAIN.

Architecture:
    raw MRI (NoNormalization plans) → EDAINv1Layer → nnU-Net backbone

EDAIN's sublayers (paper Sanna Passino et al. 2024):
    h1: outlier mitigation (alpha, beta)
    h2: shift (m)
    h3: scale (s)
    h4: power transform (lambda, OPTIONAL — see nnUNetTrainerEDAINv1Power)

CONFIGURATION VIA ENV VARS
==========================
    EDAIN_V1_STATS_JSON     path to per-case stats JSON
                            (produced by mri_edain_v1.precompute.precompute_v1_stats)
                            REQUIRED.
    EDAIN_V1_USE_POWER      "1" to enable h4 (Yeo-Johnson). Default: "0".
    EDAIN_V1_INIT_ALPHA     default 0.5
    EDAIN_V1_INIT_BETA      default 1.5
    EDAIN_V1_INIT_M         default 0.0
    EDAIN_V1_INIT_S         default 1.0
    EDAIN_V1_INIT_LAMBDA    default 1.0
    EDAIN_V1_LR_ALPHA       LR multiplier for alpha; default 10
    EDAIN_V1_LR_BETA        LR multiplier for beta;  default 10
    EDAIN_V1_LR_SHIFT       LR multiplier for m;     default 1
    EDAIN_V1_LR_SCALE       LR multiplier for s;     default 1
    EDAIN_V1_LR_POWER       LR multiplier for lambda;default 10
    EDAIN_V1_RESCALE_P2P98  "1" to enable percentile-based pre-rescale to [0,1].
                            Default "1" (recommended for raw MRI).

REQUIREMENTS
============
nnU-Net preprocessing MUST be done with the raw (NoNormalization) plans:
    nnUNetv2_preprocess -d 500 -c 3d_fullres -plans_name nnUNetPlans_raw
Run with -p nnUNetPlans_raw at training time.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mri_edain_v1.modules.edain_v1_layer import EDAINv1Layer
from mri_edain_v1.modules.percentile_stats import (
    load_stats_dict, stats_dict_to_tensor_lookup
)

from edain_nnunet.network.edain_v1_wrapper import EDAINv1Wrapper


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    return float(v)


class nnUNetTrainerEDAINv1(nnUNetTrainer):
    """nnU-Net augmented with the 4-sublayer EDAIN v1 normalization."""

    # ----- configuration from env -----
    @property
    def use_power_transform(self) -> bool:
        return _env_bool("EDAIN_V1_USE_POWER", False)

    @property
    def rescale_with_percentile(self) -> bool:
        return _env_bool("EDAIN_V1_RESCALE_P2P98", True)

    @property
    def stats_json_path(self) -> Path:
        p = os.environ.get("EDAIN_V1_STATS_JSON")
        if p is None:
            raise RuntimeError(
                "env var EDAIN_V1_STATS_JSON must point to the per-case stats JSON. "
                "Generate it via:\n"
                "  python -m mri_edain_v1.precompute.precompute_v1_stats "
                "    --preprocessed_dir <preproc>/nnUNetPlans_raw_3d_fullres "
                "    --splits_json <preproc>/splits_final.json "
                "    --fold <F> "
                "    --output_json edain_v1_stats_fold<F>.json"
            )
        return Path(p)

    @property
    def edain_lr_multipliers(self) -> Dict[str, float]:
        return {
            "alpha": _env_float("EDAIN_V1_LR_ALPHA", 10.0),
            "beta":  _env_float("EDAIN_V1_LR_BETA", 10.0),
            "shift": _env_float("EDAIN_V1_LR_SHIFT", 1.0),
            "scale": _env_float("EDAIN_V1_LR_SCALE", 1.0),
            "power": _env_float("EDAIN_V1_LR_POWER", 10.0),
        }

    # ----- override: initialize -----
    def initialize(self):
        super().initialize()

        stats_path = self.stats_json_path
        if not stats_path.exists():
            raise FileNotFoundError(
                f"stats JSON not found: {stats_path}\n"
                "Run mri_edain_v1.precompute.precompute_v1_stats first."
            )
        self.print_to_log_file(f"[EDAINv1] loading stats: {stats_path}")
        raw_stats = load_stats_dict(stats_path)
        case_stats_table = stats_dict_to_tensor_lookup(raw_stats)
        self.print_to_log_file(
            f"[EDAINv1] loaded stats for {len(case_stats_table)} cases")

        # Build EDAIN v1 layer
        edain = EDAINv1Layer(
            init_alpha=_env_float("EDAIN_V1_INIT_ALPHA", 0.5),
            init_beta=_env_float("EDAIN_V1_INIT_BETA", 1.5),
            init_m=_env_float("EDAIN_V1_INIT_M", 0.0),
            init_s=_env_float("EDAIN_V1_INIT_S", 1.0),
            init_lambda=_env_float("EDAIN_V1_INIT_LAMBDA", 1.0),
            use_power_transform=self.use_power_transform,
            rescale_with_percentile=self.rescale_with_percentile,
        )
        self.print_to_log_file(
            f"[EDAINv1] layer init: {edain.extra_repr()}")

        # Wrap the (already-built) backbone with EDAIN.
        self.network = EDAINv1Wrapper(
            edain=edain,
            backbone=self.network,
            case_stats_table=case_stats_table,
        ).to(self.device)

        # Recreate the optimizer with per-sublayer LR groups for the EDAIN params.
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.print_to_log_file(
            f"[EDAINv1] lr_multipliers={self.edain_lr_multipliers} "
            f"use_power={self.use_power_transform} "
            f"rescale_p2p98={self.rescale_with_percentile}")

    # ----- override: configure_optimizers (per-sublayer LR groups) -----
    def configure_optimizers(self):
        """Override to create separate LR groups for EDAIN sublayers.

        Backbone keeps nnU-Net's default LR (self.initial_lr, default 1e-2 SGD).
        Each EDAIN sublayer parameter uses initial_lr * lr_multiplier.
        """
        # If network is not yet wrapped (called from super().initialize() before
        # our wrapping), defer to parent.
        if not hasattr(self.network, "edain"):
            return super().configure_optimizers()

        base_lr = self.initial_lr
        multipliers = self.edain_lr_multipliers

        edain_layer = self.network.edain
        backbone = self.network.backbone

        param_groups = []
        # Per-sublayer EDAIN groups
        edain_param_ids = set()
        for name, param in edain_layer.named_sublayer_params():
            mult = multipliers.get(name, 1.0)
            param_groups.append({
                "params": [param],
                "lr": base_lr * mult,
                "name": f"edain_{name}",
                "_lr_multiplier": mult,
            })
            edain_param_ids.add(id(param))

        # Backbone group (everything else)
        backbone_params = [p for p in backbone.parameters() if id(p) not in edain_param_ids]
        param_groups.append({
            "params": backbone_params,
            "lr": base_lr,
            "name": "backbone",
            "_lr_multiplier": 1.0,
        })

        optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )

        # We reuse nnU-Net's PolyLRScheduler which expects optimizer + initial_lr.
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, base_lr, self.num_epochs)

        return optimizer, lr_scheduler

    # ----- override: train_step / validation_step to push case_ids -----
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

    # ----- override: log EDAIN parameter values each epoch -----
    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if hasattr(self.network, "edain"):
            with torch.no_grad():
                self.print_to_log_file(
                    f"[EDAINv1] {self.network.edain.extra_repr()}")
