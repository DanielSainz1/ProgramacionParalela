from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .base import BaseEvaluator


class MultiprocessingEvaluator(BaseEvaluator):
    """V2 — Evaluador multiproceso con ProcessPoolExecutor.

    Utiliza ``concurrent.futures.ProcessPoolExecutor`` y su método
    ``executor.map()`` para evaluar las partículas en procesos separados
    del sistema operativo.  Cada proceso tiene su propio intérprete
    Python y su propio GIL, lo que permite **paralelismo real** en
    múltiples núcleos de CPU.

    Batching (chunksize)
    --------------------
    En lugar de enviar 1 partícula por tarea al pool (lo que generaría
    ``n_particles`` llamadas IPC), se usa el parámetro ``chunksize`` de
    ``executor.map()`` para agrupar partículas en lotes (chunks).
    Cada lote se serializa (pickle) y se envía a un proceso worker como
    una sola unidad, reduciendo el número de transferencias IPC.

    Ejemplo: con 100 partículas y ``chunksize=10``, se crean 10 tareas
    en vez de 100, reduciendo 10x el overhead de serialización/IPC.

    Coste de IPC y serialización (pickling)
    ----------------------------------------
    Los datos (posiciones ``np.ndarray``) deben serializarse con pickle
    para transferirse entre procesos vía IPC (inter-process communication).
    Para funciones objetivo baratas (sphere, rastrigin — microsegundos),
    este coste de serialización domina y el evaluador multiproceso es
    **más lento** que el secuencial.  El beneficio aparece cuando la
    función objetivo es computacionalmente cara (simulaciones, ML), donde
    el tiempo de cómputo supera con creces el overhead de IPC.

    Parámetros
    ----------
    objective : callable
        Función objetivo (debe ser pickleable — funciones definidas a
        nivel de módulo).
    max_workers : int
        Número de procesos worker en el pool.  Configurable desde
        ``PSOConfig.n_workers``.
    chunksize : int
        Tamaño de lote para ``executor.map()``.  Configurable desde
        ``PSOConfig.chunk_size``.
    """

    def __init__(self, objective, max_workers=4, chunksize=10, **kwargs):
        self.objective = objective
        self.max_workers = max_workers
        self.chunksize = chunksize

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.objective, positions, chunksize=self.chunksize)
            return np.array(list(results))
        