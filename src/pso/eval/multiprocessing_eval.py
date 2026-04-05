from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .base import BaseEvaluator


class MultiprocessingEvaluator(BaseEvaluator):
    """V2 — Multiprocessing evaluator using ProcessPoolExecutor.

    Each worker is a separate OS process with its own Python interpreter
    and GIL, so this gives real parallelism. The chunksize parameter
    controls batching — instead of sending 1 particle per IPC call, we
    send chunks to reduce serialization (pickle) overhead.

    For cheap functions like sphere/rastrigin the IPC cost is bigger than
    the actual computation, so this ends up slower than sequential. It
    would help with expensive objectives (simulations, ML, etc).
    """

    def __init__(self, objective, max_workers=4, chunksize=10, **kwargs):
        self.objective = objective
        self.max_workers = max_workers
        self.chunksize = chunksize

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.objective, positions, chunksize=self.chunksize)
            return np.array(list(results))
        