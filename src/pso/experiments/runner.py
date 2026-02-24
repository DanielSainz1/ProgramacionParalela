import numpy as np
from .config import PSOConfig
from ..core.pso import run_pso, PSOResult
from ..eval.sequential import SequentialEvaluator
from ..objectives import OBJECTIVES

def run_pso_from_config(cfg: PSOConfig) -> PSOResult:
    #Looks for the objective function
    objective = OBJECTIVES[cfg.objective]

    lower = np.full(cfg.dim, cfg.lower)
    upper = np.full(cfg.dim, cfg.upper)

    evaluator = SequentialEvaluator(objective)

    return run_pso(objective, cfg.dim, cfg.n_particles, cfg.max_iter,
    cfg.w, cfg.c1, cfg.c2, lower, upper, evaluator, seed = cfg.seed)