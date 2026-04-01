import time
import logging
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config

logging.basicConfig(level=logging.WARNING)

EVALUATORS = ["sequential", "threading", "multiprocessing"]
OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DIMS = [2, 10, 30]

def main():
    cfg = PSOConfig.from_yaml("configs/default.yaml")

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
                print(f" {ev:20s} | {elapsed:.4f}s | best: {result.best_value:.6e}")

            base = times["sequential"]
            for ev in EVALUATORS:
                speedup = base / times[ev]
                print(f" {ev:20s} | speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
