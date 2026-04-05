import logging
import numpy as np
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    evaluator = SequentialEvaluator(sphere)

    positions = np.array([
        [1.0, -2.0, 0.5],
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
    ])

    fitness = evaluator.evaluate(positions)
    logger.info("fitness = %s", fitness)


if __name__ == "__main__":
    main()
