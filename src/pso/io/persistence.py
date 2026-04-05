import json
import csv
from pathlib import Path
from datetime import datetime

from ..experiments.config import PSOConfig
from ..core.pso import PSOResult
from .metadata import get_git_hash, get_hardware_info

def save_run(cfg: PSOConfig, result: PSOResult, out_dir: str = "results/") -> Path:
    """Save a PSO run to disk (config.json + metrics.csv in a timestamped folder)."""
    # nombre de la carpeta con timestamp para que no se pisen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{cfg.objective}_d{cfg.dim}_s{cfg.seed}"

    run_dir = Path(out_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # guardar config + tiempos + resultado + metadata
    data = {
        "pso": cfg.__dict__,
        "timing": {
            "total_time": round(result.total_time, 6),
            "eval_time": round(result.eval_time, 6),
            "update_time": round(result.update_time, 6),
            "overhead": round(result.overhead, 6),
        },
        "result": {
            "best_value": result.best_value,
            "best_position": result.best_position.tolist(),
        },
        "meta": {
            "git_hash": get_git_hash(),
            "timestamp": timestamp,
            "hardware": get_hardware_info(),
        },
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(data, f, indent=4)

    # guardar metricas por iteracion
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "gbest_value"])
        for i,val in enumerate(result.best_history):
            writer.writerow([i,val])

    return run_dir

