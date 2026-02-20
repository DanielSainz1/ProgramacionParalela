from .sphere import sphere
from .rosenbrock import rosenbrock
from .rastrigin import rastrigin
from .ackley import ackley

OBJECTIVES = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}

BOUNDS = {
    "sphere": (-100.0, 100.0),
    "rosenbrock": (-5.0, 10.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
}