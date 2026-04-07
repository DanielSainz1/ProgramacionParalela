import numpy as np
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator

D = 2
LOWER = np.full(D, -10.0)
UPPER = np.full(D, 10.0)
EVALUATOR = SequentialEvaluator(sphere)

## To make sure sphere function converges to 0.
def test_sphere_convergence():
    seed = 12345
    result = run_pso(sphere, D, 20, 100, 0.719, 1.49445, 1.49445, LOWER, UPPER, EVALUATOR, seed)
    assert result.best_value < 1e-6