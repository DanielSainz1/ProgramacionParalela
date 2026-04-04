from typing import Callable
import numpy as np

from .base import BaseEvaluator


class SequentialEvaluator(BaseEvaluator):
    """V0 — Evaluador secuencial (baseline).

    Evalúa cada partícula en un bucle for simple, sin ningún tipo de
    paralelismo.  Es la opción más rápida cuando la función objetivo es
    barata (microsegundos) porque no incurre en overhead de creación de
    hilos ni procesos, ni en serialización (pickling) de datos.
    """

    def __init__(self, objective: Callable[[np.ndarray], float], **kwargs):
        # **kwargs permite que runner.py pase max_workers/chunksize
        # sin que SequentialEvaluator se rompa.
        self.objective = objective

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        n_particles = positions.shape[0]
        fitness = np.empty(n_particles, dtype=float)

        for i in range(n_particles):
            fitness[i] = self.objective(positions[i])

        return fitness