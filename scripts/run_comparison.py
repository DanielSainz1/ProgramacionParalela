"""V0/V1/V2 comparison with multi-seed statistics.

Runs each (objective, dim, evaluator) combination across several seeds and
reports mean/std for time and speedup. This gives statistically meaningful
numbers instead of single-run measurements that depend on scheduler noise.
"""
import time
import csv
import logging
import statistics
import matplotlib.pyplot as plt
import numpy as np

from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.objectives import BOUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("pso").setLevel(logging.WARNING)  # silence PSO internal logs
logger = logging.getLogger(__name__)

EVALUATORS = ["sequential", "threading", "multiprocessing"]
OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DIMS = [2, 10, 30]
SEEDS = [42, 7, 123, 1, 99]


def run_one(cfg: PSOConfig):
    """Run a single PSO configuration and return total time + breakdown."""
    start = time.perf_counter()
    result = run_pso_from_config(cfg)
    elapsed = time.perf_counter() - start
    return elapsed, result


def main():
    cfg = PSOConfig.from_yaml("configs/default.yaml")
    rows = []

    for obj in OBJECTIVES:
        lo, hi = BOUNDS[obj]
        for dim in DIMS:
            logger.info("=== %s | dim=%d ===", obj, dim)

            # aggregate per-evaluator timings across seeds
            seed_times = {ev: [] for ev in EVALUATORS}
            seed_eval = {ev: [] for ev in EVALUATORS}
            seed_update = {ev: [] for ev in EVALUATORS}
            seed_best = {ev: [] for ev in EVALUATORS}

            for seed in SEEDS:
                for ev in EVALUATORS:
                    cfg.objective = obj
                    cfg.dim = dim
                    cfg.lower = lo
                    cfg.upper = hi
                    cfg.evaluator = ev
                    cfg.seed = seed

                    elapsed, result = run_one(cfg)
                    seed_times[ev].append(elapsed)
                    seed_eval[ev].append(result.eval_time)
                    seed_update[ev].append(result.update_time)
                    seed_best[ev].append(result.best_value)

            # aggregate stats
            mean_times = {ev: statistics.mean(seed_times[ev]) for ev in EVALUATORS}
            std_times = {ev: statistics.stdev(seed_times[ev]) for ev in EVALUATORS}
            base = mean_times["sequential"]

            for ev in EVALUATORS:
                row = {
                    "objective": obj,
                    "dim": dim,
                    "evaluator": ev,
                    "n_seeds": len(SEEDS),
                    "mean_total_time": round(mean_times[ev], 4),
                    "std_total_time": round(std_times[ev], 4),
                    "mean_eval_time": round(statistics.mean(seed_eval[ev]), 4),
                    "mean_update_time": round(statistics.mean(seed_update[ev]), 4),
                    "mean_overhead": round(
                        mean_times[ev]
                        - statistics.mean(seed_eval[ev])
                        - statistics.mean(seed_update[ev]),
                        4,
                    ),
                    "pct_eval": round(100 * statistics.mean(seed_eval[ev]) / mean_times[ev], 1),
                    "pct_update": round(100 * statistics.mean(seed_update[ev]) / mean_times[ev], 1),
                    "mean_best_value": statistics.mean(seed_best[ev]),
                    "speedup": round(base / mean_times[ev], 3),
                }
                rows.append(row)
                logger.info(
                    "  %16s | total=%.4f±%.4fs | eval=%.1f%% | update=%.1f%% | speedup=%.2fx",
                    ev, row["mean_total_time"], row["std_total_time"],
                    row["pct_eval"], row["pct_update"], row["speedup"],
                )

    # Save CSV
    csv_path = "results/comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results saved to %s", csv_path)

    # Generate speedup plot
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(16, 4), sharey=True)
    for i, obj in enumerate(OBJECTIVES):
        obj_rows = [r for r in rows if r["objective"] == obj]
        for ev in EVALUATORS:
            ev_rows = [r for r in obj_rows if r["evaluator"] == ev]
            dims = [r["dim"] for r in ev_rows]
            speedups = [r["speedup"] for r in ev_rows]
            axes[i].plot(dims, speedups, marker="o", label=ev)
        axes[i].set_title(obj)
        axes[i].set_xlabel("Dimensions")
        axes[i].set_xticks(DIMS)
        axes[i].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Speedup vs sequential")
    axes[-1].legend(loc="best")
    plt.tight_layout()
    plot_path = "results/speedup.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("Plot saved to %s", plot_path)


if __name__ == "__main__":
    main()
