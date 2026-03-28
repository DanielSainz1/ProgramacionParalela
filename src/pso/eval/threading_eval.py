from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .base import BaseEvaluator

class ThreadingEvaluator(BaseEvaluator):
    #4 different threads
    def __init__(self,objective,max_workers =4):
        self.objective = objective
        self.max_workers = max_workers

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.objective, positions)
            return np.array(list(results))


        