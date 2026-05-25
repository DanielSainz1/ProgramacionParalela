import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock Function:
        f(x) = sum_{i=0}^{d-2} [ 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2 ]

    Global minimum: f(1, 1, ..., 1) = 0
    Typical bounds: [-5, 10]
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_vec(X: np.ndarray) -> np.ndarray:
    """Vectorised Rosenbrock: X of shape (N, d) -> ndarray of shape (N,)."""
    X = np.asarray(X, dtype=float)
    xi = X[..., :-1]
    xi1 = X[..., 1:]
    return np.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1.0 - xi) ** 2, axis=-1)
