import numpy as np


def ackley(x: np.ndarray) -> float:
    """
    Ackley Function:
        f(x) = -20*exp(-0.2*sqrt(sum(x^2)/d))
               - exp(sum(cos(2*pi*x))/d)
               + 20 + e

    Global minimum: f(0, 0, ..., 0) = 0
    Typical bounds: [-32.768, 32.768]
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        - np.exp(sum_cos / d)
        + 20.0
        + np.e
    )
