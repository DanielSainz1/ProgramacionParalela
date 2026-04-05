from itertools import product
from pathlib import Path
import csv
import copy
import logging

import numpy as np

from .config import PSOConfig
from .runner import run_pso_from_config

logger = logging.getLogger(__name__)


def _auc(history: list[float]) -> float:
    """Area Under the Curve of the convergence curve (trapezoidal rule)."""
    return float(np.trapezoid(history))


def _convergence_iter(history: list[float], tol: float = 1e-10) -> int:
    """Iteration at which cumulative improvement falls below tol.

    Returns the index of the first iteration where
    history[i-1] - history[i] < tol for 10 consecutive iterations.
    If it never converges, returns len(history)-1.
    """
    streak = 0
    for i in range(1, len(history)):
        if history[i - 1] - history[i] < tol:
            streak += 1
            if streak >= 10:
                return i - 10 + 1
        else:
            streak = 0
    return len(history) - 1


def grid_search(
    base_cfg: PSOConfig,
    w_values: list,
    c1_values: list,
    c2_values: list,
    seeds: list,
    evaluators: list[str] | None = None,
    n_particles_values: list[int] | None = None,
    max_iter_values: list[int] | None = None,
    out_path: str = "results/grid_search.csv",
) -> str:
    """Run grid search over hyperparameters and save results to a CSV.
    If evaluators is None, uses the one from base_cfg.
    Returns the path to the generated CSV.
    """
    if evaluators is None:
        evaluators = [base_cfg.evaluator]
    if n_particles_values is None:
        n_particles_values = [base_cfg.n_particles]
    if max_iter_values is None:
        max_iter_values = [base_cfg.max_iter]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "evaluator", "n_particles", "max_iter", "w", "c1", "c2", "seed",
        "best_value", "total_time", "eval_time", "update_time", "overhead",
        "auc", "convergence_iter",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in evaluators:
            for np_ in n_particles_values:
                for mi in max_iter_values:
                    for w, c1, c2 in product(w_values, c1_values, c2_values):
                        for seed in seeds:
                            cfg = copy.copy(base_cfg)
                            cfg.evaluator = ev
                            cfg.n_particles = np_
                            cfg.max_iter = mi
                            cfg.w = w
                            cfg.c1 = c1
                            cfg.c2 = c2
                            cfg.seed = seed

                            result = run_pso_from_config(cfg)

                            writer.writerow({
                                "evaluator": ev,
                                "n_particles": np_,
                                "max_iter": mi,
                                "w": w,
                                "c1": c1,
                                "c2": c2,
                                "seed": seed,
                                "best_value": result.best_value,
                                "total_time": round(result.total_time, 6),
                                "eval_time": round(result.eval_time, 6),
                                "update_time": round(result.update_time, 6),
                                "overhead": round(result.overhead, 6),
                                "auc": round(_auc(result.best_history), 6),
                                "convergence_iter": _convergence_iter(result.best_history),
                            })
                            logger.debug("grid: ev=%s np=%d mi=%d w=%.3f c1=%.3f c2=%.3f seed=%d → %.4e",
                                         ev, np_, mi, w, c1, c2, seed, result.best_value)

    logger.info("Grid search completed → %s", out_path)
    return out_path
