import numpy as np
import pytest
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.objectives.ackley import ackley
from pso.objectives.rastrigin import rastrigin
from pso.objectives.rosenbrock import rosenbrock
from pso.eval.sequential import SequentialEvaluator

D = 2
LOWER = np.full(D, -10.0)
UPPER = np.full(D, 10.0)
PARAMS = dict(d=D, n_particles=20, iters=100, w=0.719, c1=1.49445, c2=1.49445,
              lower=LOWER, upper=UPPER, evaluator=SequentialEvaluator(sphere))


def test_reproducibility():
    r1 = run_pso(sphere, **PARAMS, seed=12345)
    r2 = run_pso(sphere, **PARAMS, seed=12345)
    assert r1.best_value == r2.best_value
    np.testing.assert_array_equal(r1.best_position, r2.best_position)


def test_different_seeds_give_different_results():
    """Different seeds should produce different convergence histories."""
    r1 = run_pso(sphere, **PARAMS, seed=1)
    r2 = run_pso(sphere, **PARAMS, seed=2)
    assert r1.best_history != r2.best_history


@pytest.mark.parametrize("objective", [sphere, ackley, rastrigin, rosenbrock])
def test_all_objectives_reproducible(objective):
    """Reproducibility must hold for every benchmark function."""
    ev = SequentialEvaluator(objective)
    r1 = run_pso(objective, D, 20, 100, 0.719, 1.49445, 1.49445, LOWER, UPPER, ev, seed=7)
    r2 = run_pso(objective, D, 20, 100, 0.719, 1.49445, 1.49445, LOWER, UPPER, ev, seed=7)
    assert r1.best_value == r2.best_value
