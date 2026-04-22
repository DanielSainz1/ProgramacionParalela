from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
from .base import BaseEvaluator


class ThreadingEvaluator(BaseEvaluator):
    """V1 — Threaded evaluator using ThreadPoolExecutor.

    The pool is created once in open() and reused for every evaluate() call
    during the PSO run.  close() shuts it down.  This avoids the overhead of
    spawning threads on every single iteration.

    In practice threading doesn't speed things up for our benchmark functions
    because of the GIL (Global Interpreter Lock) — Python only runs one
    thread at a time for CPU-bound code.  It would help if the objective did
    I/O or heavy NumPy operations that release the GIL.
    """

    def __init__(self, objective, max_workers=4, **kwargs):
        self.objective = objective
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None

    def open(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        if self._executor is None:
            self.open()
        return np.array(list(self._executor.map(self.objective, positions)))
