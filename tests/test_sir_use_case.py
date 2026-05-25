"""Tests for the SIR epidemic-model use case.

Three things to verify:

  1. The SIR integrator reproduces the textbook qualitative behaviour
     (curve grows, peaks, decays; total population conserved).
  2. sir_vec gives, for each particle, the exact same fitness as the
     scalar sir() — vectorisation must not change the mathematics.
  3. PSO with the runner recovers the ground-truth parameters within
     reasonable tolerance, validating the full pipeline.
"""
import numpy as np

from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.objectives import OBJECTIVES, OBJECTIVES_VEC
from pso.objectives.sir import simulate_sir, _N_POPULATION


def test_sir_integrator_qualitative_behaviour():
    """For beta > gamma the disease should grow then decay (an epidemic
    peak). For beta < gamma the disease should die out immediately."""
    growing = simulate_sir(beta=0.3, gamma=0.1, I0=10)
    dying = simulate_sir(beta=0.05, gamma=0.5, I0=10)

    # Growing run must reach a peak above the initial value
    assert growing.max() > growing[0] * 10
    # Dying run never exceeds the initial value by more than a few cases
    assert dying.max() < dying[0] * 2


def test_sir_vec_matches_scalar():
    """Per particle (in normalised [0,1]^3 space), sir_vec(Theta)[i] ==
    sir(Theta[i]) within float tolerance."""
    rng = np.random.default_rng(0)
    Theta = rng.uniform(0.0, 1.0, size=(20, 3))
    sir = OBJECTIVES["sir"]
    sir_vec = OBJECTIVES_VEC["sir"]

    expected = np.array([sir(t) for t in Theta])
    got = sir_vec(Theta)

    # The scalar version stops early when I < 1 and the vectorised one
    # keeps going. After extinction both curves are ~0 infected, so the
    # MSE is dominated by the early portion where both agree.
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-3)


def test_pso_recovers_ground_truth():
    """PSO on the SIR fitness should recover the truth (beta=0.3,
    gamma=0.1, I0=10), i.e. theta_unit = (0.3, 0.1, ~0.091), within a
    reasonable tolerance using a modest budget."""
    from pso.objectives.sir import _denormalise

    cfg = PSOConfig(
        objective="sir",
        dim=3,
        n_particles=60,
        max_iter=200,
        w=0.6,
        c1=1.5,
        c2=1.5,
        lower=0.0,
        upper=1.0,
        evaluator="vectorized",
        seed=0,
        tol=1e-12,
        stagnation=200,
        vmax_ratio=0.3,
    )
    result = run_pso_from_config(cfg)
    beta, gamma, I0 = _denormalise(result.best_position)

    assert 0.20 < beta < 0.40, f"beta out of range: {beta}"
    assert 0.05 < gamma < 0.15, f"gamma out of range: {gamma}"
    assert 1.0 < I0 < 25.0, f"I0 out of range: {I0}"
