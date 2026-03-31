from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .base import BaseEvaluator

class MultiprocessingEvaluator(BaseEvaluator):
    #4 different processes
    def __init__(self,objective,max_workers =4):
        self.objective = objective
        self.max_workers = max_workers

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.objective, positions)
            return np.array(list(results))


        