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


@pytest.mark.parametrize("objective", [sphere, ackley, rastrigin, rosenbrock])
def test_monotonic_gbest(objective):
    """Global best fitness must never increase across iterations."""
    ev = SequentialEvaluator(objective)
    result = run_pso(objective, D, 30, 200, 0.719, 1.49445, 1.49445,
                     LOWER, UPPER, ev, seed=42)
    history = np.array(result.best_history)
    for i in range(len(history) - 1):
        assert history[i + 1] <= history[i] + 1e-15, (
            f"Global best worsened at iteration {i+1}: "
            f"{history[i]:.6e} -> {history[i+1]:.6e}"
        )


def test_monotonic_gbest_high_dim():
    """Monotonicity must hold in higher dimensions too."""
    d = 10
    lower = np.full(d, -10.0)
    upper = np.full(d, 10.0)
    ev = SequentialEvaluator(sphere)
    result = run_pso(sphere, d, 40, 200, 0.719, 1.49445, 1.49445,
                     lower, upper, ev, seed=42)
    history = np.array(result.best_history)
    assert np.all(history[1:] <= history[:-1] + 1e-15)