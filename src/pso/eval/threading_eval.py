from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .base import BaseEvaluator


class ThreadingEvaluator(BaseEvaluator):
    """V1 — Evaluador multihilo con ThreadPoolExecutor.

    Utiliza ``concurrent.futures.ThreadPoolExecutor`` y su método
    ``executor.map()`` para evaluar las partículas en paralelo mediante
    hilos (threads).  Cada hilo ejecuta la función objetivo sobre un
    subconjunto de partículas.

    Cuándo mejora vs secuencial
    ---------------------------
    - **Mejora** en tareas I/O-bound (llamadas a red, lectura de ficheros)
      porque los hilos liberan el GIL durante las operaciones de espera.
    - **Mejora parcial** cuando la función objetivo usa NumPy/C internamente,
      ya que NumPy libera el GIL durante operaciones vectorizadas pesadas.
    - **No mejora** (o empeora) en código Python puro CPU-bound, porque el
      GIL (Global Interpreter Lock) impide la ejecución simultánea de
      bytecode Python: los hilos compiten por el GIL y el overhead de
      context-switching añade latencia sin beneficio.

    Parámetros
    ----------
    objective : callable
        Función objetivo que recibe un vector ``np.ndarray`` y devuelve un
        ``float``.
    max_workers : int
        Número de hilos en el pool.  Configurable desde ``PSOConfig.n_workers``.
    """

    def __init__(self, objective, max_workers=4, **kwargs):
        self.objective = objective
        self.max_workers = max_workers

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.objective, positions)
            return np.array(list(results))
