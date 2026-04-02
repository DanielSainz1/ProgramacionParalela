import time
import csv
import logging
import matplotlib.pyplot as plt
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config

logging.basicConfig(level=logging.WARNING)

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
            print(f"\n=== {obj} | dim={dim} ===")

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
                print(f"  {ev:20s} | {elapsed:.4f}s | eval={result.eval_time:.4f}s | update={result.update_time:.4f}s | best={result.best_value:.6e}")

            base = times["sequential"]
            for ev in EVALUATORS:
                speedup = base / times[ev]
                print(f"  {ev:20s} | speedup: {speedup:.2f}x")

    # Save CSV
    csv_path = "results/comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

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
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
