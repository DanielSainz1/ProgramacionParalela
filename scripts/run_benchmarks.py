from pso.experiments.config import PSOConfig
from pso.experiments.runner import run_pso_from_config
from pso.io.persistence import save_run
from pso.objectives import BOUNDS

BENCHMARK_SUITE = [
    ("sphere", [2, 10, 30]),
    ("rosenbrock", [2,10,30]),
    ("rastrigin", [2, 10, 30]),
    ("ackley", [2, 10, 30]),
]

def main():
    for objective, dims in BENCHMARK_SUITE:
        for dim in dims:
            lower, upper = BOUNDS[objective]
            cfg = PSOConfig(
                objective=objective,
                dim=dim,
                lower=lower,
                upper=upper,
                n_particles=50,
                max_iter=300,
                w=0.719,
                c1=1.49445,
                c2=1.49445,
                seed=42
            )

            result = run_pso_from_config(cfg)
            save_run(cfg, result)
            print(f"{objective:12s} d={dim:2d} best={result.best_value:.4e}")

if __name__ == "__main__":
    main()
    
