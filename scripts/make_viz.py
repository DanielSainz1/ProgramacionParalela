import argparse
import csv
import json
import logging
import numpy as np
from pathlib import Path

from pso.viz.convergence import plot_convergence
from pso.viz.swarm_animation import animate_swarm_2d
from pso.viz.swarm_3d import animate_swarm_3d
from pso.objectives import OBJECTIVES, BOUNDS
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from a saved run")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--type", type=str, default="both",
                        choices=["convergence", "animation", "both"],
                        help="Type of visualization to generate (default: both)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Read config.json to obtain dimension and function
    with open(run_dir / "config.json") as f:
        data = json.load(f)
    pso_data = data["pso"]
    objective_name = pso_data["objective"]
    dim = pso_data["dim"]

    # Generate convergence plot
    if args.type in ("convergence", "both"):
        history = []
        with open(run_dir / "metrics.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(float(row["gbest_value"]))

        plot_convergence(
            histories=[history],
            labels=[f"{objective_name} d={dim}"],
            out_path=str(run_dir / "convergence.png"),
        )
        logger.info("Convergence plot generated for %s d=%d", objective_name, dim)

    # Generate swarm animation (d=2 or d=3)
    if args.type in ("animation", "both"):
        if dim in (2, 3):
            # Re-run PSO with record_positions=True to capture particle trajectories
            cfg = PSOConfig(**pso_data)
            result = run_pso_from_config(cfg, record_positions=True)
            gbest_history = result.gbest_position_history
            lower, upper = BOUNDS[objective_name]

            if dim == 2:
                animate_swarm_2d(
                    position_history=result.position_history,
                    gbest_history=gbest_history,
                    objective=OBJECTIVES[objective_name],
                    lower=lower,
                    upper=upper,
                    out_path=str(run_dir / "swarm.gif"),
                )
            else:  # dim == 3
                animate_swarm_3d(
                    position_history=result.position_history,
                    gbest_history=gbest_history,
                    objective=OBJECTIVES[objective_name],
                    lower=lower,
                    upper=upper,
                    out_path=str(run_dir / "swarm_3d.gif"),
                )
            logger.info("Swarm animation generated for %s d=%d", objective_name, dim)
        else:
            logger.info("Skipping animation: dim=%d (only supported for dim=2 and dim=3)", dim)


if __name__ == "__main__":
    main()
