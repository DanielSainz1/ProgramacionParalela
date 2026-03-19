import argparse
from pso.experiments.config import PSOConfig
from pso.experiments.grid_search import grid_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grid Search")
    parser.add_argument("--objective", type = str, choices=["sphere", "rosenbrock", "rastrigin", "ackley"]) #define accepted arguments
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dim", type=int, default=2)
    args = parser.parse_args() #read users arguments

    cfg =  PSOConfig.from_yaml(args.config)

    if args.objective is not None:
        cfg.objective = args.objective
    if args.dim is not None:
        cfg.dim = args.dim
    
    w_values = [0.4, 0.719, 0.9]
    c1_values = [1.2, 1.49445]
    c2_values = [1.2, 1.49445]
    seeds = [42, 99, 123]
    grid_search(cfg, w_values, c1_values, c2_values, seeds)

    print("Grid search completed. Results in results/grid_search.csv")

if __name__ == "__main__":
    main()