from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Sequence
import numpy as np
from .base import BaseEvaluator


def _evaluate_batch(objective, positions_batch: List[np.ndarray]) -> List[float]:
    """Evaluate a batch of positions inside a worker process."""
    return [float(objective(x)) for x in positions_batch]


def _chunked(items: Sequence, batch_size: int) -> List[List]:
    """Split items into fixed-size batches, preserving order."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class MultiprocessingEvaluator(BaseEvaluator):
    """V2 — Multiprocessing evaluator using ProcessPoolExecutor.

    Each worker is a separate OS process with its own Python interpreter
    and GIL, so this gives real parallelism.

    The pool is created once in open() and reused across the entire PSO run.
    Positions are split into batches before being dispatched to workers, so
    each IPC round-trip carries multiple particles instead of one.  This
    reduces pickle/pipe overhead significantly.

    For cheap functions like sphere/rastrigin the IPC cost is still bigger
    than the actual computation, so V2 ends up slower than sequential.  It
    would help with expensive objectives (simulations, ML, etc).
    """

    def __init__(self, objective, max_workers=4, chunksize=10, **kwargs):
        self.objective = objective
        self.max_workers = max_workers
        self.chunksize = chunksize
        self._executor: Optional[ProcessPoolExecutor] = None

    def open(self) -> None:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        if self._executor is None:
            self.open()

        # Split positions into batches for fewer IPC round-trips
        positions_list = list(positions)
        batch_size = self.chunksize or max(1, len(positions_list) // (self.max_workers * 2))
        batches = _chunked(positions_list, batch_size)

        # Submit all batches and collect results in order
        futures = [
            self._executor.submit(_evaluate_batch, self.objective, batch)
            for batch in batches
        ]

        results: List[float] = []
        for future in futures:
            results.extend(future.result())
        return np.array(results)
        