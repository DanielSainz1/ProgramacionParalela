from typing import Callable
import numpy as np

from .base import BaseEvaluator


class SequentialEvaluator(BaseEvaluator):
    """V0 — Sequential evaluator (baseline).
    Evaluates each particle one by one in a for loop. No parallelism.
    """

    def __init__(self, objective: Callable[[np.ndarray], float], **kwargs):
        # **kwargs so runner.py can pass max_workers/chunksize without crashing
        self.objective = objective

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        n_particles = positions.shape[0]
        fitness = np.empty(n_particles, dtype=float)

        for i in range(n_particles):
            fitness[i] = self.objective(positions[i])

        return fitness