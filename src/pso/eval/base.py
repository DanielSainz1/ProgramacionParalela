from abc import ABC, abstractmethod
import numpy as np


class BaseEvaluator(ABC):
    """Base class for fitness evaluators.

    Subclasses that manage external resources (thread/process pools) should
    override open() and close().  The PSO loop calls open() once before the
    first iteration and close() once after the last iteration (via finally),
    so the pool is created once and reused across the entire run.
    """

    def open(self) -> None:
        """Allocate resources (e.g. thread/process pool). Called once per run."""

    def close(self) -> None:
        """Release resources. Called once per run (in a finally block)."""

    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate the objective function for each particle position."""
