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


def rastrigin_vec(X: np.ndarray) -> np.ndarray:
    """Vectorised Rastrigin: X of shape (N, d) -> ndarray of shape (N,)."""
    X = np.asarray(X, dtype=float)
    d = X.shape[-1]
    return 10.0 * d + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=-1)
