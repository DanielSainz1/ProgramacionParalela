import numpy as np

def sphere(x: np.ndarray) -> float:
    """
    Funcion Esgera:
        f(x) = sum_i x_i^2
    Busca minimizar.El global optimo esta en  x=0 con f(0)=0.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))