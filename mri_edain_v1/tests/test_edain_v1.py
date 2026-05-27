"""Sanity tests for EDAIN v1 layer (z-score form).

Run with:
    cd <repo root>
    python -m mri_edain_v1.tests.test_edain_v1
"""
import sys
import torch
import numpy as np

from mri_edain_v1.modules.edain_v1_layer import EDAINv1Layer
from mri_edain_v1.modules.yeo_johnson import yeo_johnson


def test_yj_identity_at_lambda_1():
    """YJ(x; 1) must equal x for all x."""
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    y = yeo_johnson(x, torch.tensor(1.0))
    assert torch.allclose(y, x, atol=1e-5), f"YJ(x; 1) != x: {y} vs {x}"
    print("[ok] YJ(x; 1) == x  (identity at init)")


def test_edain_v1_forward_shape():
    """Forward shape should match input shape."""
    layer = EDAINv1Layer(use_power_transform=False)
    B = 2
    x = torch.randn(B, 1, 8, 16, 16) * 200 + 300  # simulate raw MRI
    # fg_stats: [fg_mean, fg_std, fg_p2, fg_p98]  (p2/p98 ignored in z-score form)
    fg_stats = torch.tensor([[300.0, 200.0, 50.0, 800.0],
                              [350.0, 180.0, 60.0, 850.0]])
    y = layer(x, fg_stats)
    assert y.shape == x.shape
    print(f"[ok] forward shape preserved: {x.shape} -> {y.shape}")


def test_edain_v1_init_is_winsorized_zscore():
    """At α=1, m=1, s=1 (and h4 off), the layer must reduce to winsorized
    z-score:  y = β·tanh((x-μ)/β) / σ   (no extra raw x leaks through)."""
    # α=1 forces OM to fully replace x; β large so tanh ~ identity in central
    # range; m=1, s=1 give exact z-score affine.
    layer = EDAINv1Layer(
        init_alpha=1.0,    # full OM (no leak of raw x)
        init_beta=1e6,     # tanh ~ identity for raw-magnitude inputs
        init_m=1.0,
        init_s=1.0,
        use_power_transform=False,
    )
    rng = np.random.default_rng(2025)
    raw = rng.normal(loc=500.0, scale=200.0, size=(1, 1, 16, 32, 32)).astype(np.float32)
    raw = np.clip(raw, 1.0, 2000.0)
    x = torch.from_numpy(raw)
    fg = raw[raw > 0]
    mu, sigma = float(fg.mean()), float(fg.std())
    fg_stats = torch.tensor([[mu, sigma, 0.0, 0.0]])  # p2/p98 don't matter
    with torch.no_grad():
        y = layer(x, fg_stats)
    # Expected: y = (x - mu) / sigma  (because β is huge and α=1)
    expected = (x - mu) / sigma
    diff = (y - expected).abs().max().item()
    assert diff < 1e-3, f"z-score at init not recovered, max diff = {diff}"
    print(f"[ok] α=1, m=1, s=1, β→∞ reduces to z-score (max diff {diff:.2e})")


def test_edain_v1_default_init_centers_output():
    """With default init (α=0.5, β=800, m=1, s=1) the output should be
    centered near 0 with std near 1 for typical Lipo-scale inputs."""
    layer = EDAINv1Layer(use_power_transform=False)
    rng = np.random.default_rng(2025)
    raw = rng.normal(loc=800.0, scale=300.0, size=(1, 1, 16, 32, 32)).astype(np.float32)
    raw = np.clip(raw, 1.0, 4000.0)
    x = torch.from_numpy(raw)
    fg = raw[raw > 0]
    fg_stats = torch.tensor([[float(fg.mean()), float(fg.std()), 0.0, 0.0]])
    with torch.no_grad():
        y = layer(x, fg_stats)
    y_flat = y[x > 0]
    print(f"[ok] EDAIN v1 default init: x∈[{x.min():.0f},{x.max():.0f}], "
          f"y_mean={y_flat.mean():.3f} (≈0), y_std={y_flat.std():.3f} (≈1)")
    assert abs(y_flat.mean().item()) < 0.5, \
        f"output mean too far from 0: {y_flat.mean()}"
    assert 0.5 < y_flat.std().item() < 1.5, \
        f"output std far from 1: {y_flat.std()}"


def test_edain_v1_power_forward():
    """With power on, lambda=1 should still produce shape-preserving output."""
    layer = EDAINv1Layer(use_power_transform=True, init_lambda=1.0)
    x = torch.randn(1, 1, 8, 16, 16) * 100 + 200
    fg_stats = torch.tensor([[200.0, 100.0, 50.0, 400.0]])
    with torch.no_grad():
        y = layer(x, fg_stats)
    assert y.shape == x.shape
    print(f"[ok] EDAIN v1 + power forward: shape {y.shape}, "
          f"mean={y.mean():.3f}, std={y.std():.3f}")


def test_edain_v1_with_expanded_fg_stats():
    """REGRESSION TEST: forward must succeed when fg_stats is an EXPANDED
    tensor (stride 0 on dim 0) such as you'd get from `t.expand(B, -1)`."""
    layer = EDAINv1Layer(use_power_transform=False)
    B = 2
    x = torch.randn(B, 1, 8, 16, 16) * 100 + 200
    default_stats = torch.tensor([300.0, 200.0, 50.0, 800.0])
    fg_stats_expanded = default_stats.unsqueeze(0).expand(B, -1)
    print(f"  test_expanded: fg_stats stride={fg_stats_expanded.stride()} "
          f"(should have a 0 in it)")
    assert 0 in fg_stats_expanded.stride()
    y = layer(x, fg_stats_expanded)
    assert y.shape == x.shape
    print(f"[ok] EDAIN v1 handles expanded fg_stats")


def test_edain_v1_gradients():
    """Verify gradients flow through all 5 parameters when power is on."""
    layer = EDAINv1Layer(use_power_transform=True)
    x = torch.randn(1, 1, 4, 8, 8) * 100 + 200
    fg_stats = torch.tensor([[200.0, 100.0, 50.0, 400.0]])
    y = layer(x, fg_stats)
    loss = y.mean()
    loss.backward()
    grads = {
        "alpha": layer._alpha_raw.grad,
        "beta":  layer._beta_raw.grad,
        "m":     layer.m.grad,
        "s":     layer._s_raw.grad,
        "lambda": layer.lambda_param.grad,
    }
    for n, g in grads.items():
        assert g is not None, f"{n} has no gradient"
        if not torch.isfinite(g):
            raise AssertionError(f"{n} grad is non-finite: {g}")
    print(f"[ok] all 5 EDAIN v1 params got gradients: "
          f"{ {n: float(g) for n, g in grads.items()} }")


def main():
    print("== EDAIN v1 sanity tests ==")
    test_yj_identity_at_lambda_1()
    test_edain_v1_forward_shape()
    test_edain_v1_init_is_winsorized_zscore()
    test_edain_v1_default_init_centers_output()
    test_edain_v1_power_forward()
    test_edain_v1_with_expanded_fg_stats()
    test_edain_v1_gradients()
    print("== ALL PASSED ==")


if __name__ == "__main__":
    main()
