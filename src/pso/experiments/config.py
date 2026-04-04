from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PSOConfig:
    objective: str          # sphere, rosenbrock, rastrigin, ackley
    dim: int                # dimensiones del espacio de búsqueda
    n_particles: int
    max_iter: int
    w: float
    c1: float
    c2: float
    lower: float
    upper: float
    evaluator: str = "sequential"   # "sequential", "threading", "multiprocessing"
    seed: int = 42
    tol: float = 1e-10
    stagnation: int = 50
    n_workers: int = 4              # hilos (V1) o procesos (V2)
    chunk_size: int = 10            # tamaño de lote para V2 (batching)

    @classmethod
    def from_yaml(cls, path: str)-> PSOConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) 
        return cls(**data) 
