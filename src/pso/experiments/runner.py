import numpy as np
from .config import PSOConfig
from ..core.pso import run_pso, PSOResult
from ..eval.sequential import SequentialEvaluator
from ..objectives import OBJECTIVES, OBJECTIVES_VEC
from ..eval.threading_eval import ThreadingEvaluator
from ..eval.multiprocessing_eval import MultiprocessingEvaluator
from ..eval.async_eval import AsyncEvaluator
from ..eval.vectorized_eval import VectorizedEvaluator

EVALUATORS = {
    "sequential": SequentialEvaluator,
    "threading": ThreadingEvaluator,
    "multiprocessing": MultiprocessingEvaluator,
    "async": AsyncEvaluator,
    "vectorized": VectorizedEvaluator,
}

def run_pso_from_config(cfg: PSOConfig, record_positions: bool = False) -> PSOResult:
    # Get the objective function from the registry
    objective = OBJECTIVES[cfg.objective]

    # Creates arrays of size dim to set the bounds
    lower = np.full(cfg.dim, cfg.lower)
    upper = np.full(cfg.dim, cfg.upper)

    # First get the class, then create an instance.
    # V4 needs the vectorised version of the objective instead of the scalar one.
    evaluator_cls = EVALUATORS[cfg.evaluator]
    if cfg.evaluator == "vectorized":
        evaluator = evaluator_cls(OBJECTIVES_VEC[cfg.objective])
    else:
        evaluator = evaluator_cls(
            objective,
            max_workers=cfg.n_workers,
            chunksize=cfg.chunk_size,
        )

    return run_pso(objective, cfg.dim, cfg.n_particles, cfg.max_iter,
    cfg.w, cfg.c1, cfg.c2, lower, upper, evaluator, seed=cfg.seed,
    tol=cfg.tol, stagnation=cfg.stagnation, vmax_ratio=cfg.vmax_ratio,
    record_positions=record_positions)