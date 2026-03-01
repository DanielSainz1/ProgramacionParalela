import json
import csv
from pathlib import Path
from datetime import datetime

from ..experiments.config import PSOConfig
from ..core.pso import PSOResult
from .metadata import get_git_hash, get_hardware_info

def save_run(cfg: PSOConfig, result: PSOResult, out_dir: str = "results/") -> Path:
    #Folder's name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{cfg.objective}_d{cfg.dim}_s{cfg.seed}"

    #Create folder
    run_dir = Path(out_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    #Save config.json
    data = {
        "pso" : cfg.__dict__,
        "meta": {
            "git_hash": get_git_hash(),
            "timestamp": timestamp,
            "hardware": get_hardware_info(),
        }
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(data, f, indent = 4)

    #Save metric.csv
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "gbest_value"])
        for i,val in enumerate(result.best_history):
            writer.writerow([i,val])

    return run_dir

