"""Social topologies for PSO.

The topology decides which position each particle uses as its social
reference when updating velocity. GlobalBest (canonical PSO) uses the
swarm-wide best for every particle; local topologies (ring, von Neumann)
would restrict the reference to each particle's neighbours.

Only GlobalBestTopology is implemented in the current delivery. The ABC
exists so ring/von-Neumann variants could be plugged in without touching
the core loop.
"""
from abc import ABC, abstractmethod
import numpy as np


class Topology(ABC):
    """Abstract social topology — returns a reference position per particle."""

    @abstractmethod
    def social_best_positions(
        self,
        pbest_positions: np.ndarray,
        pbest_values: np.ndarray,
        gbest_position: np.ndarray,
    ) -> np.ndarray:
        """Return the social-best position used by each particle.

        Output shape: (n_particles, d). For a global-best topology this is
        gbest_position broadcast to every particle; for a ring topology it
        would be the best pbest among each particle's neighbours.
        """
        ...


class GlobalBestTopology(Topology):
    """Canonical gbest — every particle is attracted to the single swarm best."""

    def social_best_positions(
        self,
        pbest_positions: np.ndarray,
        pbest_values: np.ndarray,
        gbest_position: np.ndarray,
    ) -> np.ndarray:
        n = pbest_positions.shape[0]
        return np.broadcast_to(gbest_position, (n, gbest_position.shape[0])).copy()
