"""V3 — Asyncio evaluator with simulated I/O latency.

Each particle's evaluation is wrapped in a coroutine that first awaits a
short random latency (simulating a remote sensor, queued service or API
call) and then computes the objective synchronously. All N coroutines are
launched together with asyncio.gather, so their sleeps overlap on a single
thread — the event loop hands control to the next coroutine whenever one
is waiting.

This is the canonical use case for asyncio: many concurrent I/O-bound
operations that spend most of their wall time waiting. The compute itself
is single-threaded, so for a purely CPU-bound fitness with no waits this
evaluator is strictly worse than V0. It is included to show *when* asyncio
helps and when it does not.
"""
import asyncio
from typing import Callable, Optional

import numpy as np

from .base import BaseEvaluator


class AsyncEvaluator(BaseEvaluator):
    """V3 — Evaluate particles concurrently using asyncio.gather.

    A configurable random latency is awaited before each particle's
    objective call. With latency in [latency_ms_min, latency_ms_max] and
    N particles, the total wait should be ~max(latencies) instead of
    sum(latencies) thanks to cooperative concurrency.

    Set latency_ms_min = latency_ms_max = 0 to disable artificial latency
    (useful when the wrapped objective itself performs awaits, or to
    measure event-loop overhead in isolation).
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        latency_ms_min: float = 5.0,
        latency_ms_max: float = 50.0,
        latency_seed: Optional[int] = None,
        **kwargs,  # absorb max_workers / chunksize from the runner
    ):
        self.objective = objective
        self.latency_ms_min = float(latency_ms_min)
        self.latency_ms_max = float(latency_ms_max)
        self._rng = np.random.default_rng(latency_seed)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def open(self) -> None:
        """Create a dedicated event loop. Reused across all PSO iterations."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

    def close(self) -> None:
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        if self._loop is None:
            self.open()
        return self._loop.run_until_complete(self._evaluate_async(positions))

    async def _evaluate_async(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[0]
        if self.latency_ms_max > 0.0:
            latencies_s = self._rng.uniform(
                self.latency_ms_min, self.latency_ms_max, size=n
            ) / 1000.0
        else:
            latencies_s = np.zeros(n)

        coroutines = [
            self._evaluate_one(positions[i], float(latencies_s[i]))
            for i in range(n)
        ]
        results = await asyncio.gather(*coroutines)
        return np.asarray(results, dtype=float)

    async def _evaluate_one(self, x: np.ndarray, latency_s: float) -> float:
        if latency_s > 0.0:
            await asyncio.sleep(latency_s)
        return float(self.objective(x))
