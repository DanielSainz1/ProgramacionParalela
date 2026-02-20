from abc import ABC, abstractmethod
import numpy as np


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate the objective function for each particle position."""
