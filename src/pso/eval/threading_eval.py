from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .base import BaseEvaluator


class ThreadingEvaluator(BaseEvaluator):
    """V1 — Threaded evaluator using ThreadPoolExecutor.

    Uses threads to evaluate particles in parallel. In practice this
    doesn't speed things up for our benchmark functions because of the GIL
    (Global Interpreter Lock) — Python only runs one thread at a time for
    CPU-bound code. It would help if the objective did I/O or heavy NumPy
    operations that release the GIL.
    """

    def __init__(self, objective, max_workers=4, **kwargs):
        self.objective = objective
        self.max_workers = max_workers

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.objective, positions)
            return np.array(list(results))
