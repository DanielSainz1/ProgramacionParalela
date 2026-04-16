"""V2 (multiprocessing) batching experiment.

Measures how chunk_size affects the total time of V2. Small chunks mean
many IPC round-trips (pickle + pipe transfer per particle). Large chunks
amortise IPC but reduce load balancing across workers. The goal is to
find an optimal batch size empirically.

Runs a fixed objective/dimension across several seeds for statistical
validity and saves results to results/batching.csv + results/batching.png.
"""
import time
import csv
import logging
import statistics
import matplotlib.pyplot as plt

from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.objectives import BOUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("pso").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

OBJECTIVE = "ackley"     # moderately expensive, good candidate for V2
DIM = 30
N_PARTICLES = 160
MAX_ITER = 400
N_WORKERS = 4
SEEDS = [42, 7, 123]
CHUNK_SIZES = [1, 4, 8, 16, 32, 64, 128]


def measure_v2(chunk_size: int, seed: int) -> float:
    lo, hi = BOUNDS[OBJECTIVE]
    cfg = PSOConfig(
        objective=OBJECTIVE,
        dim=DIM,
        n_particles=N_PARTICLES,
        max_iter=MAX_ITER,
        w=0.719,
        c1=1.49445,
        c2=1.49445,
        lower=lo,
        upper=hi,
        evaluator="multiprocessing",
        seed=seed,
        n_workers=N_WORKERS,
        chunk_size=chunk_size,
    )
    start = time.perf_counter()
    run_pso_from_config(cfg)
    return time.perf_counter() - start


def measure_v0(seed: int) -> float:
    """Baseline V0 time for the same configuration (for speedup calculation)."""
    lo, hi = BOUNDS[OBJECTIVE]
    cfg = PSOConfig(
        objective=OBJECTIVE,
        dim=DIM,
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
    start = time.perf_counter()
    run_pso_from_config(cfg)
    return time.perf_counter() - start


def main():
    logger.info("Batching experiment: %s d=%d particles=%d iters=%d workers=%d",
                OBJECTIVE, DIM, N_PARTICLES, MAX_ITER, N_WORKERS)

    # V0 baseline for speedup calculation
    v0_times = [measure_v0(s) for s in SEEDS]
    v0_mean = statistics.mean(v0_times)
    logger.info("V0 baseline: mean=%.3fs (n=%d seeds)", v0_mean, len(SEEDS))

    rows = []
    for chunk in CHUNK_SIZES:
        times = [measure_v2(chunk, s) for s in SEEDS]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        speedup = v0_mean / mean_t
        rows.append({
            "chunk_size": chunk,
            "mean_time": round(mean_t, 3),
            "std_time": round(std_t, 3),
            "v0_mean": round(v0_mean, 3),
            "speedup_vs_v0": round(speedup, 3),
            "n_seeds": len(SEEDS),
        })
        logger.info("chunk=%3d | V2=%.3f±%.3fs | speedup vs V0=%.2fx",
                    chunk, mean_t, std_t, speedup)

    # Save CSV
    csv_path = "results/batching.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", csv_path)

    # Plot: time vs chunk_size (log x-axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    chunks = [r["chunk_size"] for r in rows]
    means = [r["mean_time"] for r in rows]
    stds = [r["std_time"] for r in rows]
    speedups = [r["speedup_vs_v0"] for r in rows]

    ax1.errorbar(chunks, means, yerr=stds, marker="o", capsize=4, label="V2")
    ax1.axhline(y=v0_mean, color="gray", linestyle="--", label=f"V0 baseline ({v0_mean:.2f}s)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("chunk_size")
    ax1.set_ylabel("Total time (s)")
    ax1.set_title(f"V2 time vs batch size ({OBJECTIVE}, d={DIM})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(chunks, speedups, marker="o", color="tab:green")
    ax2.axhline(y=1.0, color="gray", linestyle="--", label="V0 parity")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("chunk_size")
    ax2.set_ylabel("Speedup vs V0")
    ax2.set_title("Effective speedup")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "results/batching.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("Saved %s", plot_path)


if __name__ == "__main__":
    main()
