from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SwarmState:
    """Holds all the mutable state of the swarm (positions, velocities, bests)."""
    positions: np.ndarray
    velocities: np.ndarray
    pbest_positions: np.ndarray
    pbest_values: np.ndarray
    gbest_position: np.ndarray
    gbest_value: float