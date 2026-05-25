"""V4 — Vectorised NumPy evaluator (implicit parallelism).

Instead of evaluating particles one by one in a Python loop, V4 calls a
vectorised version of the objective that takes the full (N, d) position
matrix and returns the (N,) fitness array in a single NumPy operation.

There is no thread/process pool here. The speedup comes from:

1. Eliminating the Python interpreter overhead of N function calls per
   iteration.
2. NumPy dispatching the math to BLAS/SIMD instructions in C, which
   process several floats per CPU cycle (AVX/SSE).

This is the canonical demonstration that "less code" sometimes beats
"more processes". For cheap objectives (Sphere, Rastrigin) V4 is
typically the fastest of all variants.
"""
from typing import Callable

import numpy as np

from .base import BaseEvaluator


class VectorizedEvaluator(BaseEvaluator):
    """V4 — Evaluate all particles at once with a vectorised objective.

    The constructor takes an `objective_vec` callable with the contract
    `objective_vec(X: (N, d) ndarray) -> (N,) ndarray`. The mapping
    between scalar and vectorised objectives is resolved by the runner
    via the ``OBJECTIVES_VEC`` registry.
    """

    def __init__(self, objective_vec: Callable[[np.ndarray], np.ndarray], **kwargs):
        self.objective_vec = objective_vec

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.objective_vec(positions), dtype=float)
