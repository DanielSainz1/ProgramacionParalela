import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin Function:
        f(x) = 10*d + sum_{i=0}^{d-1} [ x[i]^2 - 10*cos(2*pi*x[i]) ]

    Global minimum: f(0, 0, ..., 0) = 0
    Typical bounds: [-5.12, 5.12]
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))
