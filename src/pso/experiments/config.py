from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PSOConfig:
    objective: str #sphere,rosenbrock ...
    dim : int #dimensions
    n_particles: int
    max_iter: int
    w: float
    c1: float
    c2: float
    lower: float
    upper: float
    evaluator: str = "sequential"   # "sequential", "threading", etc.
    seed: int = 42
    tol: float = 1e-10
    stagnation: int = 50

    @classmethod
    def from_yaml(cls, path: str)-> PSOConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) 
        return cls(**data) 
