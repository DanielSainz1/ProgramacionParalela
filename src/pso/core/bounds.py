import numpy as np

def clamp_positions(positions: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Clamp each coordinate of positions to [lower, upper].
    positions: (n, d)
    lower, upper: (d,)
    """
    return np.clip(positions, lower, upper)
