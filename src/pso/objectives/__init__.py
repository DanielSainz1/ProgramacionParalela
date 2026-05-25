from .sphere import sphere, sphere_vec
from .rosenbrock import rosenbrock, rosenbrock_vec
from .rastrigin import rastrigin, rastrigin_vec
from .ackley import ackley, ackley_vec

OBJECTIVES = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}

# Vectorised counterparts: each takes (N, d) and returns (N,) instead of
# being called once per particle. Used by V4 VectorizedEvaluator.
OBJECTIVES_VEC = {
    "sphere": sphere_vec,
    "rosenbrock": rosenbrock_vec,
    "rastrigin": rastrigin_vec,
    "ackley": ackley_vec,
}

BOUNDS = {
    "sphere": (-100.0, 100.0),
    "rosenbrock": (-5.0, 10.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
}