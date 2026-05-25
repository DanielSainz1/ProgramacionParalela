"""Tests for the on_iteration callback hook."""
import numpy as np
from pso.core.pso import run_pso
from pso.core.state import SwarmState
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator

D = 2
LOWER = np.full(D, -10.0)
UPPER = np.full(D, 10.0)


def _run(on_iteration=None, iters=50):
    ev = SequentialEvaluator(sphere)
    return run_pso(sphere, D, 10, iters, 0.719, 1.49445, 1.49445,
                   LOWER, UPPER, ev, seed=42, stagnation=iters,
                   on_iteration=on_iteration)


def test_callback_called_every_iteration():
    """Callback must be called once per iteration with (int, SwarmState)."""
    calls = []
    def recorder(it, state):
        calls.append((it, type(state)))
    _run(on_iteration=recorder, iters=30)
    assert len(calls) == 30
    assert all(it == i for i, (it, _) in enumerate(calls))
    assert all(t is SwarmState for _, t in calls)


def test_callback_sees_updated_gbest():
    """The state passed to the callback must reflect the current gbest."""
    gbest_log = []
    def track_gbest(it, state):
        gbest_log.append(state.gbest_value)
    _run(on_iteration=track_gbest, iters=50)
    # gbest must be monotonically non-increasing
    for i in range(len(gbest_log) - 1):
        assert gbest_log[i + 1] <= gbest_log[i] + 1e-15


def test_no_callback_is_fine():
    """on_iteration=None must not cause any errors."""
    result = _run(on_iteration=None)
    assert result.best_value < 1.0
