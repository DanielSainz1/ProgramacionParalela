"""V3 latency experiment — when does asyncio actually pay off?

Compares V0 (sequential) against V3 (async with simulated latency) on
Sphere d=10 across a sweep of mean latency values (1, 5, 10, 20, 50 ms
per particle). The cheap-fitness comparison in run_comparison.py
already shows V3 is *not* worth it when there is nothing to wait for;
this script shows the inverse — that V3 wins by orders of magnitude
once the workload includes I/O-style latency.

Produces results/latency.csv and results/latency.png.
"""
import csv
import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from pso.core.pso import run_pso
from pso.eval.async_eval import AsyncEvaluator
from pso.eval.sequential import SequentialEvaluator
from pso.objectives import OBJECTIVES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("pso").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Fixed configuration to isolate the effect of latency
OBJECTIVE = "sphere"
DIM = 10
N_PARTICLES = 20                            # small swarm: each iter spends N*latency_ms in sleep
ITERS = 15                                  # short, so the total run stays under ~5 min
SEEDS = [0, 1, 2]
LATENCIES_MS = [1, 5, 10, 20, 50]


def _evaluator_with_latency(name: str, latency_ms: float, seed: int):
    obj = OBJECTIVES[OBJECTIVE]
    if name == "sequential":
        # Wrap sequential in a function that sleeps before computing,
        # so V0 and V3 face the same per-particle latency budget.
        def slow_obj(x, base=obj, lat=latency_ms / 1000.0):
            time.sleep(lat)
            return base(x)
        return SequentialEvaluator(slow_obj)
    elif name == "async":
        return AsyncEvaluator(
            obj,
            latency_ms_min=latency_ms,
            latency_ms_max=latency_ms,
            latency_seed=seed,
        )
    raise ValueError(name)


def _run_one(evaluator_name: str, latency_ms: float, seed: int) -> float:
    lower = np.full(DIM, -5.0)
    upper = np.full(DIM, 5.0)
    ev = _evaluator_with_latency(evaluator_name, latency_ms, seed)
    t0 = time.perf_counter()
    run_pso(
        OBJECTIVES[OBJECTIVE], DIM, N_PARTICLES, ITERS,
        0.719, 1.49445, 1.49445,
        lower, upper, ev, seed=seed, stagnation=ITERS,
    )
    return time.perf_counter() - t0


def main():
    rows = []
    for lat in LATENCIES_MS:
        for name in ["sequential", "async"]:
            seed_times = [_run_one(name, lat, s) for s in SEEDS]
            mean_t = float(np.mean(seed_times))
            std_t = float(np.std(seed_times))
            rows.append({
                "latency_ms": lat,
                "evaluator": name,
                "mean_time": round(mean_t, 3),
                "std_time": round(std_t, 3),
                "n_seeds": len(SEEDS),
            })
            logger.info("lat=%2d ms | %10s | %.3f ± %.3f s", lat, name, mean_t, std_t)

    # CSV
    csv_path = "results/latency.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ["sequential", "async"]:
        xs = [r["latency_ms"] for r in rows if r["evaluator"] == name]
        ys = [r["mean_time"] for r in rows if r["evaluator"] == name]
        ax.plot(xs, ys, marker="o", label=f"V0 ({name})" if name == "sequential" else "V3 (async)")
    ax.set_xlabel("Simulated latency per particle (ms)")
    ax.set_ylabel("Total wall time (s)")
    ax.set_title(f"V0 vs V3 — Sphere d={DIM}, {N_PARTICLES} particles, {ITERS} iters")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = "results/latency.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    logger.info("Saved %s", plot_path)


if __name__ == "__main__":
    main()
