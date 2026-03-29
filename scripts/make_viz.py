import argparse
import csv
import json
import numpy as np
from pathlib import Path

from pso.viz.convergence import plot_convergence
from pso.viz.swarm_animation import animate_swarm_2d
from pso.objectives import OBJECTIVES, BOUNDS
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from a saved run")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    #Read config.json to obtain dimension and function
    with open(run_dir / "config.json") as f:
        data = json.load(f)
    pso_data = data["pso"]
    objective_name = pso_data["objective"]
    dim = pso_data["dim"]

    # Read metrics.csv to obtain convergence history
    history = []
    with open(run_dir / "metrics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(float(row["gbest_value"]))
    
    #Generate graphic to see convergence
    plot_convergence(
        histories=[history],
        labels=[f"{objective_name} d={dim}"],
        out_path=str(run_dir / "convergence.png"),
    )

    #Confirm it's 2d and then create animation
    if dim==2:
        # Re-run PSO with record_positions = True
        cfg = PSOConfig(**pso_data)
        result = run_pso_from_config(cfg, record_positions=True)
        gbest_history = result.gbest_position_history
        lower, upper= BOUNDS[objective_name]
        animate_swarm_2d(
            position_history=result.position_history,
            gbest_history=gbest_history,
            objective=OBJECTIVES[objective_name],
            lower=lower,
            upper=upper,
            out_path=str(run_dir / "swarm.gif"),
        )

if __name__ == "__main__":
    main()
