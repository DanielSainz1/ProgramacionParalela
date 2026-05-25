"""Use case — compare all 5 evaluators on the SIR calibration problem.

This is the script that exercises the parallelism strategies on a *real*
fitness function rather than on a microsecond synthetic benchmark. Each
evaluation integrates a sub-day Euler step of the SIR ODE across 100
days and computes the MSE against the observed daily-infected curve in
data/sir_observations.csv.

Per-particle scalar cost is in the ~1 ms range, and with a swarm of 100
particles each iteration spends ~100 ms in fitness — which is exactly the
regime where multiprocessing (V2) and vectorisation (V4) should beat V0.

Produces results/sir_comparison.csv (one row per evaluator) and prints a
recovery summary showing how close each strategy got to the ground truth
(beta=0.30, gamma=0.10, I0=10).
"""
import csv
import logging
import statistics
import time

from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.objectives.sir import _denormalise

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("pso").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

EVALUATORS = ["sequential", "threading", "multiprocessing", "async", "vectorized"]
SEEDS = [0, 1, 2]
N_PARTICLES = 100
MAX_ITER = 100


def main():
    rows = []

    for ev in EVALUATORS:
        seed_times = []
        seed_betas = []
        seed_gammas = []
        seed_I0s = []
        seed_bests = []

        for seed in SEEDS:
            cfg = PSOConfig(
                objective="sir",
                dim=3,
                n_particles=N_PARTICLES,
                max_iter=MAX_ITER,
                w=0.6,
                c1=1.5,
                c2=1.5,
                lower=0.0,
                upper=1.0,
                evaluator=ev,
                seed=seed,
                tol=1e-12,
                stagnation=MAX_ITER,        # no early stopping — comparable wall time
                vmax_ratio=0.3,
            )
            start = time.perf_counter()
            result = run_pso_from_config(cfg)
            elapsed = time.perf_counter() - start

            beta, gamma, I0 = _denormalise(result.best_position)
            seed_times.append(elapsed)
            seed_betas.append(beta)
            seed_gammas.append(gamma)
            seed_I0s.append(I0)
            seed_bests.append(result.best_value)

        mean_t = statistics.mean(seed_times)
        std_t = statistics.stdev(seed_times)
        row = {
            "evaluator": ev,
            "n_seeds": len(SEEDS),
            "mean_total_time": round(mean_t, 3),
            "std_total_time": round(std_t, 3),
            "mean_best_fitness": float(statistics.mean(seed_bests)),
            "beta": round(statistics.mean(seed_betas), 4),
            "gamma": round(statistics.mean(seed_gammas), 4),
            "I0": round(statistics.mean(seed_I0s), 3),
        }
        rows.append(row)
        logger.info(
            "%16s | %.3f ± %.3f s | beta=%.3f gamma=%.3f I0=%.2f | mse=%.2e",
            ev, mean_t, std_t, row["beta"], row["gamma"], row["I0"], row["mean_best_fitness"],
        )

    # Speedup column relative to sequential
    base = next(r["mean_total_time"] for r in rows if r["evaluator"] == "sequential")
    for r in rows:
        r["speedup_vs_v0"] = round(base / r["mean_total_time"], 3)

    csv_path = "results/sir_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", csv_path)


if __name__ == "__main__":
    main()
