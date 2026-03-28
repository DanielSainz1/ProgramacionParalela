import numpy as np
from .config import PSOConfig
from ..core.pso import run_pso, PSOResult
from ..eval.sequential import SequentialEvaluator
from ..objectives import OBJECTIVES
from ..eval.threading_eval import ThreadingEvaluator

# Faltan por añadir en V1,v2,v3,v4:
# "threading":      ThreadingEvaluator,
# "multiprocessing": ProcessEvaluator,
# "asyncio":        AsyncEvaluator,
# "vectorized":     VectorizedEvaluator,


EVALUATORS = {
    "sequential": SequentialEvaluator,
    "threading": ThreadingEvaluator,
}

def run_pso_from_config(cfg: PSOConfig, record_positions: bool = False) -> PSOResult:
    #Get the objective function from the registry
    objective = OBJECTIVES[cfg.objective]

    #Creates arrays of size dim to set the bounds
    lower = np.full(cfg.dim, cfg.lower)
    upper = np.full(cfg.dim, cfg.upper)

    #First get's the class and then makes an object of the class
    evaluator_cls = EVALUATORS[cfg.evaluator]
    evaluator = evaluator_cls(objective)

    return run_pso(objective, cfg.dim, cfg.n_particles, cfg.max_iter,
    cfg.w, cfg.c1, cfg.c2, lower, upper, evaluator, seed=cfg.seed,
    record_positions=record_positions)