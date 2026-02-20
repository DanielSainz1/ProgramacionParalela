import numpy as np
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator
from pso.core.pso import run_pso

def main() -> None:
    objective = sphere
    d = 30
    n_particles = 100
    iters = 200
    w = 0.719
    c1 = 1.49445
    c2 = 1.49445
    lower = np.full(d, -100.0)
    upper = np.full(d, 100.0)
    seed = 42

    evaluator = SequentialEvaluator(objective)
    fitness = evaluator.evaluate(np.array([[0.0] * d]))
    print("Initial fitness at [0,...,0]:", fitness[0])

    result = run_pso(sphere, d, n_particles, iters, w, c1, c2, lower, upper, evaluator, seed)
    print("Best value:", result.best_value)
    print("Best position (first 5 dims):", result.best_position[:5])
    print("Last 5 history values:", result.best_history[-5:])


if __name__ == "__main__":
    main()
