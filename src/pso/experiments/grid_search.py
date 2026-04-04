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
    """Area Under the Curve de la curva de convergencia (regla del trapecio)."""
    return float(np.trapz(history))


def _convergence_iter(history: list[float], tol: float = 1e-10) -> int:
    """Iteración en la que la mejora acumulada cae por debajo de tol.

    Devuelve el índice de la primera iteración donde
    history[i-1] - history[i] < tol durante 10 iteraciones seguidas.
    Si nunca converge, devuelve len(history)-1.
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
    out_path: str = "results/grid_search.csv",
) -> str:
    """Ejecuta grid search sobre hiperparámetros y guarda resultados en CSV.

    Parameters
    ----------
    evaluators : list[str] | None
        Lista de evaluadores a probar (e.g. ["sequential", "threading"]).
        Si es None, usa el evaluador de base_cfg.

    Returns
    -------
    str
        Ruta al CSV generado.
    """
    if evaluators is None:
        evaluators = [base_cfg.evaluator]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "evaluator", "w", "c1", "c2", "seed",
        "best_value", "total_time", "eval_time", "update_time", "overhead",
        "auc", "convergence_iter",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in evaluators:
            for w, c1, c2 in product(w_values, c1_values, c2_values):
                for seed in seeds:
                    cfg = copy.copy(base_cfg)
                    cfg.evaluator = ev
                    cfg.w = w
                    cfg.c1 = c1
                    cfg.c2 = c2
                    cfg.seed = seed

                    result = run_pso_from_config(cfg)

                    writer.writerow({
                        "evaluator": ev,
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
                    logger.debug("grid: ev=%s w=%.3f c1=%.3f c2=%.3f seed=%d → %.4e",
                                 ev, w, c1, c2, seed, result.best_value)

    logger.info("Grid search completado → %s", out_path)
    return out_path
