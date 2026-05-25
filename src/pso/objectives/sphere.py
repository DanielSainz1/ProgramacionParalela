import numpy as np

def sphere(x: np.ndarray) -> float:
    """
    Sphere Function:
        f(x) = sum_i x_i^2
    Searches for the minimum. Global minimum is at x=0 with f(0)=0.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def sphere_vec(X: np.ndarray) -> np.ndarray:
    """Vectorised sphere: X of shape (N, d) -> ndarray of shape (N,)."""
    X = np.asarray(X, dtype=float)
    return np.sum(X ** 2, axis=-1)