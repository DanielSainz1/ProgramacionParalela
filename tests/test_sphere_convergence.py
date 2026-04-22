import numpy as np
import pytest
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator


def _run_sphere(d, n_particles=40, iters=500, seed=42, stagnation=80):
    lower = np.full(d, -10.0)
    upper = np.full(d, 10.0)
    return run_pso(sphere, d, n_particles, iters, 0.719, 1.49445, 1.49445,
                   lower, upper, SequentialEvaluator(sphere), seed=seed,
                   stagnation=stagnation)


def test_sphere_convergence_d2():
    result = _run_sphere(d=2)
    assert result.best_value < 1e-6


def test_sphere_convergence_d10():
    result = _run_sphere(d=10, n_particles=60)
    assert result.best_value < 1e-4


def test_sphere_convergence_d30():
    result = _run_sphere(d=30, n_particles=100, stagnation=100)
    assert result.best_value < 1.0


def test_sphere_best_position_near_origin():
    """Best position must be close to the true optimum (all zeros)."""
    result = _run_sphere(d=2)
    np.testing.assert_allclose(result.best_position, np.zeros(2), atol=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_sphere_converges_across_seeds(seed):
    """Sphere must converge regardless of seed."""
    result = _run_sphere(d=2, seed=seed)
    assert result.best_value < 1e-5