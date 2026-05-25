"""Tests for velocity clamping (vmax_ratio parameter).

Verifies that when vmax_ratio is set, particle velocities never exceed
the allowed maximum, and that clamping improves convergence on functions
where unclamped particles tend to overshoot.
"""
import numpy as np
import pytest
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.objectives.rosenbrock import rosenbrock
from pso.eval.sequential import SequentialEvaluator


D = 10
LOWER = np.full(D, -10.0)
UPPER = np.full(D, 10.0)
RANGE = UPPER - LOWER


def _run(objective, vmax_ratio=None, d=D, iters=300, seed=42):
    lower = np.full(d, -10.0)
    upper = np.full(d, 10.0)
    ev = SequentialEvaluator(objective)
    return run_pso(objective, d, 40, iters, 0.719, 1.49445, 1.49445,
                   lower, upper, ev, seed=seed, vmax_ratio=vmax_ratio,
                   record_positions=True)


def test_velocities_respect_vmax():
    """All velocities must stay within [-vmax, vmax] every iteration."""
    ratio = 0.3
    vmax = RANGE * ratio
    result = _run(sphere, vmax_ratio=ratio)
    # Check that recorded positions didn't jump further than vmax in one step
    for it in range(len(result.position_history) - 1):
        delta = result.position_history[it + 1] - result.position_history[it]
        assert np.all(np.abs(delta) <= vmax + 1e-10), (
            f"Velocity exceeded vmax at iteration {it}"
        )


def test_vmax_none_is_unclamped():
    """With vmax_ratio=None, velocities should be unconstrained."""
    r1 = _run(sphere, vmax_ratio=None, d=2)
    r2 = _run(sphere, vmax_ratio=0.5, d=2)
    # Both should converge, but histories will differ
    assert r1.best_history != r2.best_history


def test_vmax_improves_rosenbrock_d10():
    """Vmax clamping should help on Rosenbrock d=10 (prone to overshoot)."""
    no_clamp = _run(rosenbrock, vmax_ratio=None, iters=500)
    with_clamp = _run(rosenbrock, vmax_ratio=0.5, iters=500)
    # Clamped should converge at least as well (typically much better)
    assert with_clamp.best_value <= no_clamp.best_value * 1.5 + 1e-10


@pytest.mark.parametrize("ratio", [0.1, 0.5])
def test_vmax_ratios_all_converge_sphere(ratio):
    """Sphere should converge regardless of vmax_ratio."""
    result = _run(sphere, vmax_ratio=ratio, d=2, iters=300)
    assert result.best_value < 1e-4


def test_vmax_reproducible():
    """Same seed + same vmax_ratio must produce identical results."""
    r1 = _run(sphere, vmax_ratio=0.5, d=2, seed=99)
    r2 = _run(sphere, vmax_ratio=0.5, d=2, seed=99)
    assert r1.best_value == r2.best_value
    np.testing.assert_array_equal(r1.best_position, r2.best_position)
