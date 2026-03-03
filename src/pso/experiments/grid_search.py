from itertools import product
from pathlib import Path
import csv
import copy

from .config import PSOConfig
from .runner import run_pso_from_config


def grid_search(
    base_cfg: PSOConfig,
    w_values: list,
    c1_values: list,
    c2_values: list,
    seeds: list,
    out_path: str = "results/grid_search.csv",
) -> None:

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w", "c1", "c2", "seed", "best_value"])

        for w, c1, c2 in product(w_values, c1_values, c2_values):
            for seed in seeds:
                cfg = copy.copy(base_cfg)
                cfg.w = w
                cfg.c1 = c1
                cfg.c2 = c2
                cfg.seed = seed
                result = run_pso_from_config(cfg)
                writer.writerow([w, c1, c2, seed, result.best_value])
