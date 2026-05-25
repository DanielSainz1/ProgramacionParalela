"""Boundary handling strategies for PSO.

Defines an abstract BoundsPolicy contract so different strategies (clamp,
reflect, penalty) can be swapped without touching the core PSO loop. Only
ClampBounds is implemented in the current delivery — it clips positions
back into the search box and zeroes the corresponding velocity component
to prevent the particle from immediately trying to escape again.

The free function clamp_positions() is kept for convenience and is used
internally by ClampBounds.
"""
from abc import ABC, abstractmethod
import numpy as np


def clamp_positions(positions: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clamp each coordinate of positions to [lower, upper].

    positions: (n, d)
    lower, upper: (d,)
    """
    return np.clip(positions, lower, upper)


class BoundsPolicy(ABC):
    """Abstract boundary-handling policy applied after each position update."""

    @abstractmethod
    def apply(self, positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Enforce the search-box constraint.

        Returns (positions, velocities) after correcting any particle that
        left the feasible region. Implementations may or may not modify
        velocities — clamping zeroes velocity components that hit the wall
        to avoid oscillation, reflection would flip their sign, etc.
        """
        ...


class ClampBounds(BoundsPolicy):
    """Clip positions to [lower, upper]. Zero velocity on hit axes.

    Zeroing the velocity component prevents the particle from immediately
    trying to exit again on the next iteration; the cognitive and social
    terms then redirect it back into the interior.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def apply(self, positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(positions, self.lower, self.upper)
        hit_wall = clipped != positions
        velocities = np.where(hit_wall, 0.0, velocities)
        return clipped, velocities


class ReflectBounds(BoundsPolicy):
    """Reflect positions off the boundary like a billiard ball.

    When a particle crosses a wall, the excess distance is mirrored back
    into the feasible region and the velocity component is flipped.  This
    conserves kinetic energy, whereas ClampBounds kills it.  The result is
    better exploration near corners of the search space at the cost of
    slightly less stable convergence near the optimum.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self._range = self.upper - self.lower

    def apply(self, positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Normalize to [0, 1] range, reflect using triangle-wave, then scale back
        normalized = (positions - self.lower) / self._range
        # Triangle wave: fold into [0, 1] regardless of how far out of bounds
        folded = np.abs(normalized % 2.0 - 1.0)
        # Convert back: 0 maps to upper, 1 maps to lower after abs trick,
        # so we use (1 - folded) to keep the correct orientation
        reflected = self.lower + (1.0 - folded) * self._range
        # Flip velocity on axes that were out of bounds
        below = positions < self.lower
        above = positions > self.upper
        velocities = np.where(below | above, -velocities, velocities)
        return reflected, velocities
