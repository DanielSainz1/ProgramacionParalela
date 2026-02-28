import argparse
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSO")
    parser.add_argument("--objective", type = str, choices=["sphere", "rosenbrock", "rastrigin", "ackley"]) #define accepted arguments
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args() #read users arguments

    cfg =  PSOConfig.from_yaml(args.config)

    if args.objective is not None:
        cfg.objective = args.objective
    if args.dim is not None:
        cfg.dim = args.dim
    if args.seed is not None:
        cfg.seed = args.seed

    result = run_pso_from_config(cfg)
    print("Objective: ", cfg.objective, "| Dim: ", cfg.dim, " | Seed: ", cfg.seed)
    print("Best value: ",  result.best_value)
    print("Best position (first 5 dimensions): ", result.best_position[:5])


if __name__ == "__main__":
    main()
