import numpy as np
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator

D = 2
LOWER = np.full(D, -10.0)
UPPER = np.full(D, 10.0)
EVALUATOR = SequentialEvaluator(sphere)

## Make sure best_history doesn't go up.
def test_monotonic_gbest():
    seed = 12345
    result = run_pso(sphere,D,20,100,0.719,1.49445,1.49445,LOWER,UPPER,EVALUATOR,seed)
    history = np.array(result.best_history)
    assert np.all(history[1:] <= history[:-1])