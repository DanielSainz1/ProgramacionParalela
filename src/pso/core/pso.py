from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
import logging
import time

from .bounds import BoundsPolicy, ClampBounds
from .topology import Topology, GlobalBestTopology
from .state import SwarmState
from ..eval.base import BaseEvaluator

logger = logging.getLogger(__name__)

@dataclass
class PSOResult:
    """Stores the result of a PSO run: best solution, convergence history and timing."""
    best_position: np.ndarray
    best_value: float
    best_history: List[float]
    position_history: list
    gbest_position_history: list
    total_time: float
    eval_time: float
    update_time: float
    overhead: float          # total - eval - update (swarm management, logging, etc.)


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
    tol: float = 1e-10,
    stagnation: int = 50,
    record_positions: bool = False,
    bounds_policy: Optional[BoundsPolicy] = None,
    topology: Optional[Topology] = None,
) -> PSOResult:
    """Run the PSO algorithm.

    The evaluator, bounds policy and topology are all injected so the same
    loop works for V0/V1/V2, any bounds strategy (clamp, reflect, penalty)
    and any topology (gbest, ring, von Neumann). Defaults are ClampBounds
    and GlobalBestTopology when not provided.

    Returns a PSOResult with the best solution found, the convergence
    history and a timing breakdown (eval, update, overhead).
    """
    if bounds_policy is None:
        bounds_policy = ClampBounds(lower, upper)
    if topology is None:
        topology = GlobalBestTopology()

    rng = np.random.default_rng(seed)
    logger.info("PSO start: %d particles, dim=%d, iters=%d", n_particles, d, iters)
    positions = rng.uniform(lower, upper, size=(n_particles, d))
    velocities = rng.uniform((-0.1 * (upper - lower)), (0.1 * (upper - lower)), size=(n_particles, d))
    positions, velocities = bounds_policy.apply(positions, velocities)

    t_start = time.perf_counter()
    t_eval = 0.0
    t_update = 0.0

    t0 = time.perf_counter()
    fitness = evaluator.evaluate(positions)
    t_eval += time.perf_counter() - t0

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=positions.copy(),
        pbest_values=fitness.copy(),
        gbest_position=positions[fitness.argmin()].copy(),
        gbest_value=float(fitness.min()),
    )

    best_history = [state.gbest_value]
    position_history = []
    gbest_position_history = []
    no_improve = 0  # stagnation counter

    for it in range(iters):
        r1 = rng.random((n_particles, d))
        r2 = rng.random((n_particles, d))

        t0 = time.perf_counter()
        social_best = topology.social_best_positions(
            state.pbest_positions, state.pbest_values, state.gbest_position
        )
        state.velocities = (
            w * state.velocities
            + c1 * r1 * (state.pbest_positions - state.positions)
            + c2 * r2 * (social_best - state.positions)
        )

        state.positions += state.velocities
        state.positions, state.velocities = bounds_policy.apply(state.positions, state.velocities)
        t_update += time.perf_counter() - t0

        if record_positions:
            position_history.append(state.positions.copy())

        t0 = time.perf_counter()
        fitness = evaluator.evaluate(state.positions)
        t_eval += time.perf_counter() - t0

        improved_mask = fitness < state.pbest_values
        state.pbest_positions[improved_mask] = state.positions[improved_mask]
        state.pbest_values[improved_mask] = fitness[improved_mask]

        gbest_index = np.argmin(state.pbest_values)
        if state.pbest_values[gbest_index] < state.gbest_value:
            state.gbest_position = state.pbest_positions[gbest_index].copy()
            state.gbest_value = float(state.pbest_values[gbest_index])

        if best_history[-1] - state.gbest_value < tol:
            no_improve += 1
        else:
            no_improve = 0

        if no_improve >= stagnation:
            logger.info("Stopped: no improvement for %d iterations", stagnation)
            break

        if it % 50 == 0 or it == iters - 1:
            logger.info("Iter %4d | best=%.6e", it, state.gbest_value)

        best_history.append(state.gbest_value)
        gbest_position_history.append(state.gbest_position.copy())

    logger.info("PSO done: best=%.6e", state.gbest_value)

    total_time = time.perf_counter() - t_start
    overhead = total_time - t_eval - t_update
    logger.info("Times: total=%.4fs, eval=%.4fs, update=%.4fs, overhead=%.4fs",
                total_time, t_eval, t_update, overhead)

    return PSOResult(
        best_position=state.gbest_position,
        best_value=state.gbest_value,
        best_history=best_history,
        position_history=position_history,
        gbest_position_history=gbest_position_history,
        total_time=total_time,
        eval_time=t_eval,
        update_time=t_update,
        overhead=overhead,
    )
