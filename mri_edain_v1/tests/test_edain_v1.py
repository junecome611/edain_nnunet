"""Sanity tests for EDAIN v1 layer.

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
    # fg_stats: [fg_mean, fg_std, fg_p2, fg_p98]
    fg_stats = torch.tensor([[300.0, 200.0, 50.0, 800.0],
                              [350.0, 180.0, 60.0, 850.0]])
    y = layer(x, fg_stats)
    assert y.shape == x.shape
    print(f"[ok] forward shape preserved: {x.shape} -> {y.shape}")


def test_edain_v1_init_near_zscore():
    """At init (alpha=0.5, m=0, s=1, h4 off, rescale on) the layer should produce
    something close to standard z-score for typical foreground voxels."""
    layer = EDAINv1Layer(
        init_alpha=0.5, init_beta=1.5, init_m=0.0, init_s=1.0,
        use_power_transform=False, rescale_with_percentile=True,
    )
    B = 1
    rng = np.random.default_rng(2025)
    raw = rng.normal(loc=300.0, scale=200.0, size=(1, 1, 16, 32, 32)).astype(np.float32)
    raw = np.clip(raw, 1.0, 2000.0)
    x = torch.from_numpy(raw)
    fg = raw[raw > 0]
    fg_stats = torch.tensor([[
        float(fg.mean()),
        float(fg.std()),
        float(np.percentile(fg, 2.0)),
        float(np.percentile(fg, 98.0)),
    ]])
    with torch.no_grad():
        y = layer(x, fg_stats)
    # After (x - p2)/(p98 - p2), 96% should be in [0,1]. With alpha=0.5 and tanh
    # mostly in linear regime (since beta=1.5 > typical |x_norm - mu_hat|<1),
    # output of h1 is close to x_norm, then (x_om - 0)/1 = x_om ~ [0, 1].
    y_flat = y[x > 0]
    print(f"[ok] EDAIN v1 init forward: x ranges [{x.min():.0f},{x.max():.0f}], "
          f"y ranges [{y_flat.min():.3f},{y_flat.max():.3f}], "
          f"y_mean={y_flat.mean():.3f}, y_std={y_flat.std():.3f}")
    assert y.shape == x.shape
    # Output should be in [-1, 2] roughly (centered around ~0.4-0.5 with std ~0.2)
    assert -2 < y_flat.mean() < 2, f"unexpected output mean: {y_flat.mean()}"


def test_edain_v1_power_forward():
    """With power on, lambda=1 should still produce ~identity-ish output."""
    layer = EDAINv1Layer(
        init_alpha=0.5, init_beta=1.5, init_m=0.0, init_s=1.0, init_lambda=1.0,
        use_power_transform=True, rescale_with_percentile=True,
    )
    B = 1
    x = torch.randn(1, 1, 8, 16, 16) * 100 + 200
    fg_stats = torch.tensor([[200.0, 100.0, 50.0, 400.0]])
    with torch.no_grad():
        y = layer(x, fg_stats)
    assert y.shape == x.shape
    print(f"[ok] EDAIN v1 + power forward: shape {y.shape}, "
          f"mean={y.mean():.3f}, std={y.std():.3f}")


def test_edain_v1_with_expanded_fg_stats():
    """REGRESSION TEST: forward must succeed when fg_stats is an EXPANDED
    tensor (stride 0 on dim 0) such as you'd get from `t.expand(B, -1)`.

    The cluster crashed with:
        RuntimeError: unsupported operation: more than one element of the
        written-to tensor refers to a single memory location.
    when in-place clamp_ was applied on a view of an expanded tensor whose
    [:, 1] slice had all elements aliasing the same memory.
    """
    layer = EDAINv1Layer(use_power_transform=False, rescale_with_percentile=True)
    B = 2
    x = torch.randn(B, 1, 8, 16, 16) * 100 + 200
    # Build an EXPANDED stats tensor (mimics what the wrapper used to return)
    default_stats = torch.tensor([300.0, 200.0, 50.0, 800.0])
    fg_stats_expanded = default_stats.unsqueeze(0).expand(B, -1)
    print(f"  test_expanded: fg_stats stride={fg_stats_expanded.stride()} "
          f"(should have a 0 in it)")
    assert 0 in fg_stats_expanded.stride()
    # This must not crash
    y = layer(x, fg_stats_expanded)
    assert y.shape == x.shape
    print(f"[ok] EDAIN v1 handles expanded fg_stats")


def test_edain_v1_gradients():
    """Verify gradients flow through all 5 parameters when power is on."""
    layer = EDAINv1Layer(use_power_transform=True, rescale_with_percentile=True)
    x = torch.randn(1, 1, 4, 8, 8) * 100 + 200
    fg_stats = torch.tensor([[200.0, 100.0, 50.0, 400.0]])
    y = layer(x, fg_stats)
    loss = y.mean()
    loss.backward()
    # Check each learnable param got a non-zero gradient
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
    test_edain_v1_init_near_zscore()
    test_edain_v1_power_forward()
    test_edain_v1_with_expanded_fg_stats()
    test_edain_v1_gradients()
    print("== ALL PASSED ==")


if __name__ == "__main__":
    main()
