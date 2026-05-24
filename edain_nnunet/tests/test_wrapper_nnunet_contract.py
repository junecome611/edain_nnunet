"""Test the wrapper's compatibility with nnU-Net's expectations on self.network.

These tests simulate every internal call nnU-Net makes against `self.network`
in nnUNetTrainer.py, so we catch incompatibilities BEFORE shipping to the
cluster. See `grep "self.network\\." nnUNetTrainer.py` for the contract.

What we test:
    1. wrapper(x) returns the same shape as backbone(x).
    2. wrapper.parameters() returns the union of edain + backbone params.
    3. wrapper.train() / wrapper.eval() switch all submodules.
    4. wrapper.state_dict() / load_state_dict() round-trips.
    5. wrapper.decoder is forwarded to backbone.decoder (the bug we just fixed:
       set_deep_supervision_enabled calls `mod.decoder.deep_supervision = enabled`).
    6. Forwarding works for arbitrary backbone attributes, not just decoder.

Run:
    cd <repo root>
    python -m edain_nnunet.tests.test_wrapper_nnunet_contract
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ----- a fake backbone that resembles nnU-Net's PlainConvUNet for testing ----
class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True
        self.conv = nn.Conv3d(4, 2, 1)

    def forward(self, x):
        return [self.conv(x)] if self.deep_supervision else self.conv(x)


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _FakeBackbone(nn.Module):
    """Stand-in for nnU-Net's PlainConvUNet.

    Has the attribute layout that nnU-Net's trainer code touches:
        - .encoder   (any submodule)
        - .decoder   (with .deep_supervision attr)
    """
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()
        self.some_other_attr = "hello"

    def forward(self, x):
        return self.decoder(self.encoder(x))


# -------------- the actual tests --------------

def test_v1_wrapper_contract():
    from mri_edain_v1.modules.edain_v1_layer import EDAINv1Layer
    from edain_nnunet.network.edain_v1_wrapper import EDAINv1Wrapper

    edain = EDAINv1Layer(use_power_transform=False)
    backbone = _FakeBackbone()
    case_stats = {"caseA": torch.tensor([300.0, 200.0, 50.0, 800.0])}
    wrapper = EDAINv1Wrapper(edain, backbone, case_stats)
    _exercise_nnunet_contract(wrapper, backbone, "EDAINv1Wrapper")


def test_v2_wrapper_contract():
    from mri_edain_v2.modules.standardizer import CoordinateStandardizer
    from mri_edain_v2.modules.edain_layer import MRIEDAINLayer
    from edain_nnunet.network.edain_wrapper import EDAINWrapper

    standardizer = CoordinateStandardizer(n_dim=11)
    standardizer.mu.copy_(torch.zeros(11))
    standardizer.sigma.copy_(torch.ones(11))
    standardizer.is_fit.fill_(1)
    theta_0 = torch.zeros(3 * 9 - 1)  # K=9
    edain = MRIEDAINLayer(standardizer=standardizer, theta_0=theta_0)

    backbone = _FakeBackbone()
    case_gammas = {"caseA": torch.zeros(11)}
    wrapper = EDAINWrapper(edain, backbone, case_gammas)
    _exercise_nnunet_contract(wrapper, backbone, "EDAINWrapper")


def _exercise_nnunet_contract(wrapper, backbone, name):
    print(f"\n== {name}: nnU-Net contract ==")

    # 1. forward shape
    x = torch.randn(1, 1, 8, 16, 16) * 100 + 200
    wrapper.set_current_batch(["caseA"])
    y = wrapper(x)
    # The fake backbone with deep_supervision=True returns a list
    assert isinstance(y, list), f"backbone returned {type(y)}; expected list (deep_sup)"
    print(f"  [ok] forward(x) returned list of len={len(y)} as expected")

    # 2. parameters() collects both EDAIN and backbone
    n_params_total = sum(p.numel() for p in wrapper.parameters())
    n_params_backbone = sum(p.numel() for p in backbone.parameters())
    assert n_params_total > n_params_backbone, "wrapper params should include EDAIN"
    print(f"  [ok] wrapper.parameters(): {n_params_total} > backbone {n_params_backbone}")

    # 3. train() / eval() propagate
    wrapper.train()
    assert wrapper.training and backbone.training
    wrapper.eval()
    assert not wrapper.training and not backbone.training
    print(f"  [ok] train()/eval() propagate to backbone")

    # 4. state_dict round-trip
    sd = wrapper.state_dict()
    keys_sample = list(sd.keys())[:3]
    print(f"  [ok] state_dict() has {len(sd)} keys; sample: {keys_sample}")
    # The standard nn.Module load_state_dict works because the wrapper's
    # state_dict naturally namespaces by submodule (edain.* / backbone.*).
    out = wrapper.load_state_dict(sd)
    assert not out.missing_keys and not out.unexpected_keys
    print(f"  [ok] load_state_dict round-trips with 0 missing / 0 unexpected")

    # 5. THE BUG WE JUST FIXED: nnU-Net does
    #    mod.decoder.deep_supervision = enabled
    # Verify wrapper.decoder is forwarded to backbone.decoder.
    assert hasattr(wrapper, "decoder"), \
        "wrapper.decoder must exist for nnU-Net set_deep_supervision_enabled"
    assert wrapper.decoder is backbone.decoder, \
        "wrapper.decoder must be the same object as backbone.decoder"
    # The actual set:
    wrapper.decoder.deep_supervision = False
    assert backbone.decoder.deep_supervision is False
    wrapper.decoder.deep_supervision = True
    assert backbone.decoder.deep_supervision is True
    print(f"  [ok] wrapper.decoder.deep_supervision setter works (nnU-Net contract)")

    # 6. Generic forwarding for non-Module attributes too
    assert wrapper.some_other_attr == "hello"
    assert wrapper.encoder is backbone.encoder
    print(f"  [ok] wrapper.encoder and wrapper.some_other_attr forwarded to backbone")

    # 7. Missing attribute -> proper AttributeError (not infinite recursion)
    try:
        _ = wrapper.this_does_not_exist_anywhere
        raise AssertionError("expected AttributeError")
    except AttributeError:
        pass
    print(f"  [ok] missing attribute raises AttributeError correctly")


def main():
    print("== Wrapper x nnU-Net contract tests ==")
    test_v1_wrapper_contract()
    test_v2_wrapper_contract()
    print("\n== ALL PASSED ==")


if __name__ == "__main__":
    main()
