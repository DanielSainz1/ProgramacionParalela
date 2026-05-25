"""Social topologies for PSO.

The topology decides which position each particle uses as its social
reference when updating velocity. Two implementations are provided:

- GlobalBestTopology (gbest): every particle sees the swarm-wide best.
  Fast convergence, prone to premature collapse on multi-modal functions.
- RingTopology (lbest): each particle sees only its k nearest neighbours
  in a logical ring. Slower convergence but better diversity preservation.
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


class RingTopology(Topology):
    """Ring (lbest) — each particle sees only its k nearest neighbours.

    Particles are arranged in a logical ring by index.  Each particle's
    social reference is the best pbest among itself and its k neighbours
    on each side (2k+1 particles in total).  With k=1 this is the classic
    lbest-2 topology.

    Slower convergence than gbest, but much better at preserving diversity
    on multi-modal functions (Rastrigin, Ackley) where gbest collapses the
    swarm prematurely.
    """

    def __init__(self, k: int = 1) -> None:
        self.k = k

    def social_best_positions(
        self,
        pbest_positions: np.ndarray,
        pbest_values: np.ndarray,
        gbest_position: np.ndarray,
    ) -> np.ndarray:
        n = pbest_positions.shape[0]
        result = np.empty_like(pbest_positions)
        for i in range(n):
            # Indices of neighbours in the ring (wraps around)
            indices = [(i + offset) % n for offset in range(-self.k, self.k + 1)]
            best_idx = indices[int(np.argmin(pbest_values[indices]))]
            result[i] = pbest_positions[best_idx]
        return result
