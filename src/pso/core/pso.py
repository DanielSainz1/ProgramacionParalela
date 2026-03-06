from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np

from .bounds import clamp_positions
from .state import SwarmState
from ..eval.base import BaseEvaluator

@dataclass
class PSOResult:
    best_position: np.ndarray
    best_value: float
    best_history: List[float]
    position_history: list


def run_pso(objective: Callable[[np.ndarray], float],
    d: int,
    n_particles: int,
    iters: int,
    w: float,
    c1: float,
    c2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluator: BaseEvaluator,
    seed: int = 0,
    record_positions: bool = False,) -> PSOResult:

    
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lower, upper, size=(n_particles, d))
    positions = clamp_positions(positions, lower, upper)
    velocities = rng.uniform((-0.1 * (upper-lower)), (0.1 * (upper-lower)), size=(n_particles, d))

    fitness = evaluator.evaluate(positions)

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=positions.copy(),
        pbest_values=fitness.copy(),
        gbest_position=positions[np.argmin(fitness)].copy(),
        gbest_value=float(fitness[np.argmin(fitness)]),
    )

    best_history = [state.gbest_value]
    position_history = []

    for _ in range(iters):
        r1 = rng.random((n_particles, d))
        r2 = rng.random((n_particles, d))

        state.velocities = (
            w * state.velocities
            + c1 * r1 * (state.pbest_positions - state.positions)
            + c2 * r2 * (state.gbest_position - state.positions)
        )
        
        state.positions += state.velocities
        state.positions = clamp_positions(state.positions, lower, upper)

        if record_positions:
            position_history.append(state.positions.copy())

        fitness = evaluator.evaluate(state.positions)

        improved_mask = fitness < state.pbest_values
        state.pbest_positions[improved_mask] = state.positions[improved_mask]
        state.pbest_values[improved_mask] = fitness[improved_mask]

        gbest_index = np.argmin(state.pbest_values)
        if state.pbest_values[gbest_index] < state.gbest_value:
            state.gbest_position = state.pbest_positions[gbest_index].copy()
            state.gbest_value = float(state.pbest_values[gbest_index])

        best_history.append(state.gbest_value)

    return PSOResult(
        best_position=state.gbest_position,
        best_value=state.gbest_value,
        best_history=best_history,
        position_history=position_history,
    )
