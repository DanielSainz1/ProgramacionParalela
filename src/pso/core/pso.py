from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np

from .bounds import clamp_positions
from ..eval.sequential import SequentialEvaluator

@dataclass
class PSOResult:
    best_position: np.ndarray
    best_value: float
    best_history: List[float]


def run_pso(objective: Callable[[np.ndarray], float],
    d: int,
    n_particles: int,
    iters: int,
    w: float,
    c1: float,
    c2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: int = 0,) -> PSOResult:

    rng = np.random.default_rng(seed)
    positions = rng.uniform(lower, upper, size=(n_particles, d))
    positions = clamp_positions(positions, lower, upper)
    velocities = rng.uniform((-0.1 * (upper-lower)), (0.1 * (upper-lower)), size=(n_particles, d))

    evaluator = SequentialEvaluator(objective)
    fitness = evaluator.evaluate(positions) 
    pbest_positions = positions.copy()
    pbest_values = fitness.copy()
    gbest_index = np.argmin(pbest_values)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_value = float(pbest_values[gbest_index])
    best_history = [gbest_value]

    for _ in range(iters):
        r1 = rng.random((n_particles, d))
        r2 = rng.random((n_particles, d))

        velocities = w * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (gbest_position - positions)
        positions += velocities
        positions = clamp_positions(positions, lower, upper)

        fitness = evaluator.evaluate(positions)

        improved_mask = fitness < pbest_values
        pbest_positions[improved_mask] = positions[improved_mask]
        pbest_values[improved_mask] = fitness[improved_mask]

        gbest_index = np.argmin(pbest_values)
        if pbest_values[gbest_index] < gbest_value:
            gbest_position = pbest_positions[gbest_index].copy()
            gbest_value = float(pbest_values[gbest_index])

        best_history.append(gbest_value)



