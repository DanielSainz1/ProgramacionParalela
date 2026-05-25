"""Tests for V4 — VectorizedEvaluator and the *_vec objective variants.

Three properties to verify:

  1. The vectorised objective gives exactly the same result, particle by
     particle, as the scalar version. If this breaks, comparing V4 to
     V0-V3 is meaningless.
  2. The VectorizedEvaluator returns a (N,) array with the right dtype.
  3. PSO with the vectorised evaluator converges on Sphere — sanity
     check that the runner wiring picks up the right callable.
"""
import numpy as np
import pytest

from pso.core.pso import run_pso
from pso.eval.sequential import SequentialEvaluator
from pso.eval.vectorized_eval import VectorizedEvaluator
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.objectives import OBJECTIVES, OBJECTIVES_VEC


@pytest.mark.parametrize("name", ["sphere", "rosenbrock", "rastrigin", "ackley"])
def test_vectorised_matches_scalar(name):
    """For each particle, scalar(x_i) == vectorised(X)[i] to machine precision."""
    rng = np.random.default_rng(123)
    X = rng.uniform(-3.0, 3.0, size=(50, 8))

    scalar = OBJECTIVES[name]
    vec = OBJECTIVES_VEC[name]

    expected = np.array([scalar(x) for x in X])
    got = vec(X)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_evaluator_returns_correct_shape_and_dtype():
    """evaluate() must return a float ndarray with shape (N,)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(17, 4))
    ev = VectorizedEvaluator(OBJECTIVES_VEC["sphere"])
    out = ev.evaluate(X)
    assert out.shape == (17,)
    assert out.dtype == np.float64


def test_pso_converges_on_sphere_with_v4():
    """End-to-end: run_pso_from_config with evaluator=vectorized must
    converge to ~0 on Sphere d=10 with default Clerc-Kennedy params."""
    cfg = PSOConfig(
        objective="sphere",
        dim=10,
        n_particles=40,
        max_iter=300,
        w=0.719,
        c1=1.49445,
        c2=1.49445,
        lower=-5.0,
        upper=5.0,
        evaluator="vectorized",
        seed=42,
    )
    result = run_pso_from_config(cfg)
    assert result.best_value < 1e-4, f"V4 did not converge: best={result.best_value}"


def test_v4_matches_v0_numerically():
    """For the same seed, V4 and V0 should produce identical convergence
    history (vectorisation must not change the mathematics)."""
    rng_seed = 7
    d = 5
    lower = np.full(d, -5.0)
    upper = np.full(d, 5.0)

    v0 = SequentialEvaluator(OBJECTIVES["sphere"])
    v4 = VectorizedEvaluator(OBJECTIVES_VEC["sphere"])

    r0 = run_pso(OBJECTIVES["sphere"], d, 30, 50, 0.719, 1.49445, 1.49445,
                 lower, upper, v0, seed=rng_seed, stagnation=100)
    r4 = run_pso(OBJECTIVES["sphere"], d, 30, 50, 0.719, 1.49445, 1.49445,
                 lower, upper, v4, seed=rng_seed, stagnation=100)

    np.testing.assert_allclose(r0.best_history, r4.best_history, rtol=1e-12, atol=1e-12)
