"""External baseline comparison against the PySwarms library.

Runs our PSO (V0) and the canonical global-best PSO from PySwarms on the
same objectives, dimensions and seeds. Reports mean solution quality so
we can check that our implementation is competitive with a mature library.

PySwarms uses the same global-best topology and the same inertia / cognitive /
social update rule, so the comparison is fair. Our implementation can differ
slightly because of boundary handling (we clamp + zero velocity, they clamp)
and velocity initialisation (we use ±10% of range, they sample inside the
whole box).
"""
import logging
import statistics
import csv

import numpy as np
import pyswarms as ps

from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.objectives import OBJECTIVES, BOUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("pso").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

FUNCS = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DIMS = [2, 10, 30]
SEEDS = [42, 7, 123]
N_PARTICLES = 100
MAX_ITER = 500


def run_ours(obj_name: str, dim: int, seed: int) -> float:
    lo, hi = BOUNDS[obj_name]
    cfg = PSOConfig(
        objective=obj_name,
        dim=dim,
        n_particles=N_PARTICLES,
        max_iter=MAX_ITER,
        w=0.719,
        c1=1.49445,
        c2=1.49445,
        lower=lo,
        upper=hi,
        evaluator="sequential",
        seed=seed,
    )
    return run_pso_from_config(cfg).best_value


def run_pyswarms(obj_name: str, dim: int, seed: int) -> float:
    """Run PySwarms global-best PSO with matched hyperparameters."""
    np.random.seed(seed)
    lo, hi = BOUNDS[obj_name]
    bounds = (np.full(dim, lo), np.full(dim, hi))
    options = {"w": 0.719, "c1": 1.49445, "c2": 1.49445}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=N_PARTICLES, dimensions=dim, options=options, bounds=bounds,
    )
    # PySwarms expects a batch objective: (n_particles, d) -> (n_particles,)
    f = OBJECTIVES[obj_name]

    def batch_f(x):
        return np.apply_along_axis(f, 1, x)

    best_cost, _ = optimizer.optimize(batch_f, iters=MAX_ITER, verbose=False)
    return float(best_cost)


def main():
    rows = []
    for obj in FUNCS:
        for dim in DIMS:
            ours = [run_ours(obj, dim, s) for s in SEEDS]
            theirs = [run_pyswarms(obj, dim, s) for s in SEEDS]
            row = {
                "objective": obj,
                "dim": dim,
                "n_seeds": len(SEEDS),
                "ours_mean": statistics.mean(ours),
                "ours_median": statistics.median(ours),
                "pyswarms_mean": statistics.mean(theirs),
                "pyswarms_median": statistics.median(theirs),
                "winner": "ours" if statistics.median(ours) < statistics.median(theirs) else "pyswarms",
            }
            rows.append(row)
            logger.info(
                "%12s d=%2d | ours=%.3e | pyswarms=%.3e | winner=%s",
                obj, dim, row["ours_median"], row["pyswarms_median"], row["winner"],
            )

    out = "results/pyswarms_baseline.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
