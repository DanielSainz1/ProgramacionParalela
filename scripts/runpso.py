import numpy as np
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator

def main() -> None:
    evaluator = SequentialEvaluator(sphere)

    positions = np.array([
        [1.0, -2.0, 0.5],
        [0.0, 0.0, 0.0],
        [2.0,2.0,2.0]
    ])

    fitness = evaluator.evaluate(positions)
    print("fitness =", fitness)

if __name__ == "__main__":
    main()
