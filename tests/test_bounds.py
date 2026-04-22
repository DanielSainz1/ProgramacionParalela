import numpy as np
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator


def _run_and_check_bounds(d, lo, hi, seed=42):
    lower = np.full(d, lo)
    upper = np.full(d, hi)
    ev = SequentialEvaluator(sphere)
    result = run_pso(sphere, d, 30, 100, 0.719, 1.49445, 1.49445,
                     lower, upper, ev, seed=seed, record_positions=True)
    # Check final best position
    assert np.all(result.best_position >= lower), "best_position below lower bound"
    assert np.all(result.best_position <= upper), "best_position above upper bound"
    # Check ALL recorded positions across all iterations
    for it, positions in enumerate(result.position_history):
        assert np.all(positions >= lower), f"Particle below lower bound at iter {it}"
        assert np.all(positions <= upper), f"Particle above upper bound at iter {it}"


def test_bounds_d2():
    _run_and_check_bounds(d=2, lo=-10.0, hi=10.0)


def test_bounds_d10():
    _run_and_check_bounds(d=10, lo=-5.12, hi=5.12)


def test_bounds_asymmetric():
    """Bounds don't have to be symmetric around zero."""
    _run_and_check_bounds(d=2, lo=-2.0, hi=8.0, seed=0)