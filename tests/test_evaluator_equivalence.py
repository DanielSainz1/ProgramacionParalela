import numpy as np
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator
from pso.eval.threading_eval import ThreadingEvaluator
from pso.eval.multiprocessing_eval import MultiprocessingEvaluator

D = 2
LOWER = np.full(D, -10.0)
UPPER = np.full(D, 10.0)
SEED = 42
PARAMS = dict(
    objective=sphere, d=D, n_particles=30, iters=100,
    w=0.719, c1=1.49445, c2=1.49445,
    lower=LOWER, upper=UPPER, seed=SEED,
)


def test_v0_v1_same_result():
    """V0 (sequential) and V1 (threading) must produce identical results."""
    r0 = run_pso(**PARAMS, evaluator=SequentialEvaluator(sphere))
    r1 = run_pso(**PARAMS, evaluator=ThreadingEvaluator(sphere, max_workers=2))
    assert r0.best_value == r1.best_value
    np.testing.assert_array_equal(r0.best_position, r1.best_position)


def test_v0_v2_comparable_result():
    """V0 (sequential) and V2 (multiprocessing) must produce equivalent results.

    executor.map() preserves order, so results should be identical.
    We use a small tolerance as a safety net.
    """
    r0 = run_pso(**PARAMS, evaluator=SequentialEvaluator(sphere))
    r2 = run_pso(**PARAMS, evaluator=MultiprocessingEvaluator(sphere, max_workers=2, chunksize=5))
    np.testing.assert_allclose(r0.best_value, r2.best_value, atol=1e-12)
    np.testing.assert_allclose(r0.best_position, r2.best_position, atol=1e-12)
