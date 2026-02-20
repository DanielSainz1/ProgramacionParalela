import numpy as np

def sphere(x: np.ndarray) -> float:
    """
    Sphere Function:
        f(x) = sum_i x_i^2
    Searches for the minimum .Best global is in x=0 with f(0)=0.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))