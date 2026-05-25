"""Tests for V3 — AsyncEvaluator.

Verifies the three properties that matter:
  1. Fitness values match the sequential evaluator (latency does not change
     the objective, only the wall time).
  2. The open()/close() lifecycle is reusable.
  3. asyncio.gather actually overlaps the simulated latencies — total wall
     time should be close to max(latency), not sum(latency).
"""
import time

import numpy as np
import pytest

from pso.eval.async_eval import AsyncEvaluator
from pso.eval.sequential import SequentialEvaluator
from pso.objectives.sphere import sphere


@pytest.fixture
def positions():
    rng = np.random.default_rng(42)
    return rng.uniform(-5.0, 5.0, size=(20, 4))


def test_async_returns_same_values_as_sequential(positions):
    """Async wrapping must not change the fitness — only the wall time."""
    seq = SequentialEvaluator(sphere)
    async_ev = AsyncEvaluator(sphere, latency_ms_min=0.0, latency_ms_max=0.0)
    async_ev.open()
    try:
        expected = seq.evaluate(positions)
        got = async_ev.evaluate(positions)
        np.testing.assert_allclose(got, expected, atol=1e-12)
    finally:
        async_ev.close()


def test_open_close_is_reusable():
    """Multiple open/close cycles must not leak event loops."""
    ev = AsyncEvaluator(sphere, latency_ms_min=0.0, latency_ms_max=0.0)
    for _ in range(3):
        ev.open()
        ev.evaluate(np.zeros((5, 2)))
        ev.close()
    assert ev._loop is None


def test_gather_overlaps_latencies():
    """With N particles each waiting ~30 ms, total time must be much less
    than 30 ms * N. If it is not, asyncio.gather is not actually
    concurrent. We allow a generous margin for jitter/CI noise.
    """
    n = 16
    latency_ms = 30.0
    positions = np.zeros((n, 2))
    ev = AsyncEvaluator(
        sphere,
        latency_ms_min=latency_ms,
        latency_ms_max=latency_ms,
        latency_seed=0,
    )
    ev.open()
    try:
        t0 = time.perf_counter()
        ev.evaluate(positions)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        ev.close()

    # Serial lower bound: n * latency_ms = 480 ms.
    # Concurrent upper bound: ~latency_ms + event-loop overhead.
    assert elapsed_ms < 0.25 * n * latency_ms, (
        f"Async did not overlap latencies: {elapsed_ms:.1f} ms for "
        f"{n} x {latency_ms:.0f} ms tasks (serial would be {n*latency_ms:.0f} ms)"
    )
