import time
import csv
import logging
import matplotlib.pyplot as plt
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EVALUATORS = ["sequential", "threading", "multiprocessing"]
OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DIMS = [2, 10, 30]

def main():
    cfg = PSOConfig.from_yaml("configs/default.yaml")
    rows = []

    for obj in OBJECTIVES:
        for dim in DIMS:
            cfg.objective = obj
            cfg.dim = dim
            logger.warning("=== %s | dim=%d ===", obj, dim)

            times = {}
            for ev in EVALUATORS:
                cfg.evaluator = ev
                start = time.perf_counter()
                result = run_pso_from_config(cfg)
                elapsed = time.perf_counter() - start

                times[ev] = elapsed
                speedup = times["sequential"] / elapsed if "sequential" in times else 1.0

                row = {
                    "objective": obj,
                    "dim": dim,
                    "evaluator": ev,
                    "total_time": round(elapsed, 4),
                    "eval_time": round(result.eval_time, 4),
                    "update_time": round(result.update_time, 4),
                    "overhead": round(elapsed - result.eval_time - result.update_time, 4),
                    "best_value": result.best_value,
                    "speedup": round(speedup, 2),
                }
                rows.append(row)
                logger.warning("  %20s | %.4fs | eval=%.4fs | update=%.4fs | best=%.6e", ev, elapsed, result.eval_time, result.update_time, result.best_value)

            base = times["sequential"]
            for ev in EVALUATORS:
                speedup = base / times[ev]
                logger.warning("  %20s | speedup: %.2fx", ev, speedup)

    # Save CSV
    csv_path = "results/comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.warning("Results saved to %s", csv_path)

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
    logger.warning("Plot saved to %s", plot_path)

if __name__ == "__main__":
    main()
