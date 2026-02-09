from typing import Callable
import numpy as np

class SequentialEvaluator:
    def __init__(self, objective: Callable[[np.ndarray], float]):
        self.objective = objective

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        n_particles = positions.shape[0]
        fitness = np.empty(n_particles, dtype=float)

        for i in range(n_particles):
            fitness[i] = self.objective(positions[i])
        
        return fitness