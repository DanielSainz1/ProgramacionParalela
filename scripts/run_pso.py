import argparse
from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSO")
    parser.add_argument("--objective", type=str, choices=["sphere", "rosenbrock", "rastrigin", "ackley"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluator", type=str)
    parser.add_argument("--n-workers", type=int, help="hilos (V1) o procesos (V2)")
    parser.add_argument("--chunk-size", type=int, help="batch size para V2")
    parser.add_argument("--profile", action="store_true", help="ejecutar con cProfile")
    args = parser.parse_args()

    cfg = PSOConfig.from_yaml(args.config)

    # sobreescribir los campos que se pasen por CLI
    if args.objective is not None:
        cfg.objective = args.objective
    if args.dim is not None:
        cfg.dim = args.dim
    if args.seed is not None:
        cfg.seed = args.seed
    if args.evaluator is not None:
        cfg.evaluator = args.evaluator
    if args.n_workers is not None:
        cfg.n_workers = args.n_workers
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size

    if args.profile:
        import cProfile, pstats, io as _io
        pr = cProfile.Profile()
        pr.enable()
        result = run_pso_from_config(cfg)
        pr.disable()
        s = _io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
        logging.getLogger(__name__).info("cProfile report:\n%s", s.getvalue())
    else:
        result = run_pso_from_config(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Objective: %s | Dim: %d | Seed: %d", cfg.objective, cfg.dim, cfg.seed)
    logger.info("Best value: %.6e", result.best_value)
    logger.info("Best position (first 5 dims): %s", result.best_position[:5])


if __name__ == "__main__":
    main()
