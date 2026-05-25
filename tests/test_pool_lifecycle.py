"""Tests for evaluator pool lifecycle (open / close).

Verifies that ThreadingEvaluator and MultiprocessingEvaluator create their
pool in open(), reuse it across multiple evaluate() calls, and release
resources in close().
"""
import numpy as np
import pytest

from pso.objectives.sphere import sphere
from pso.eval.threading_eval import ThreadingEvaluator
from pso.eval.multiprocessing_eval import MultiprocessingEvaluator
from pso.eval.sequential import SequentialEvaluator


POSITIONS = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
EXPECTED = np.array([sphere(p) for p in POSITIONS])


def test_threading_open_creates_pool():
    ev = ThreadingEvaluator(sphere, max_workers=2)
    assert ev._executor is None
    ev.open()
    assert ev._executor is not None
    ev.close()
    assert ev._executor is None


def test_threading_reuses_pool_across_calls():
    ev = ThreadingEvaluator(sphere, max_workers=2)
    ev.open()
    pool_id = id(ev._executor)
    ev.evaluate(POSITIONS)
    ev.evaluate(POSITIONS)
    assert id(ev._executor) == pool_id  # same pool object
    ev.close()


def test_multiprocessing_open_creates_pool():
    ev = MultiprocessingEvaluator(sphere, max_workers=2)
    assert ev._executor is None
    ev.open()
    assert ev._executor is not None
    ev.close()
    assert ev._executor is None


def test_multiprocessing_reuses_pool_across_calls():
    ev = MultiprocessingEvaluator(sphere, max_workers=2, chunksize=2)
    ev.open()
    pool_id = id(ev._executor)
    ev.evaluate(POSITIONS)
    ev.evaluate(POSITIONS)
    assert id(ev._executor) == pool_id
    ev.close()


def test_sequential_open_close_are_harmless():
    """SequentialEvaluator should accept open/close without errors."""
    ev = SequentialEvaluator(sphere)
    ev.open()
    result = ev.evaluate(POSITIONS)
    np.testing.assert_allclose(result, EXPECTED)
    ev.close()


def test_multiprocessing_rejects_lambda():
    """Lambdas can't be pickled — open() must fail fast with a clear message."""
    ev = MultiprocessingEvaluator(lambda x: float(x @ x), max_workers=2)
    with pytest.raises(TypeError, match="cannot be pickled"):
        ev.open()


def test_multiprocessing_rejects_closure():
    """Closures can't be pickled either."""
    offset = 5.0
    def closed_over(x):
        return float(x @ x) + offset
    ev = MultiprocessingEvaluator(closed_over, max_workers=2)
    with pytest.raises(TypeError, match="cannot be pickled"):
        ev.open()
