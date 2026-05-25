from .sphere import sphere, sphere_vec
from .rosenbrock import rosenbrock, rosenbrock_vec
from .rastrigin import rastrigin, rastrigin_vec
from .ackley import ackley, ackley_vec
from .sir import sir, sir_vec

OBJECTIVES = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "sir": sir,
}

# Vectorised counterparts: each takes (N, d) and returns (N,) instead of
# being called once per particle. Used by V4 VectorizedEvaluator.
OBJECTIVES_VEC = {
    "sphere": sphere_vec,
    "rosenbrock": rosenbrock_vec,
    "rastrigin": rastrigin_vec,
    "ackley": ackley_vec,
    "sir": sir_vec,
}

BOUNDS = {
    "sphere": (-100.0, 100.0),
    "rosenbrock": (-5.0, 10.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    # SIR works in a normalised [0, 1]^3 cube: the fitness rescales each
    # coordinate back to physical units (beta, gamma in [0, 1]; I0 in
    # [1, 100]). This avoids per-dimension bounds while keeping a
    # well-conditioned uniform search box.
    "sir": (0.0, 1.0),
}