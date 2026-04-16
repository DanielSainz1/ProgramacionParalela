import numpy as np
from pso.core.bounds import ClampBounds


def test_clamp_bounds_clips_positions():
    """Positions outside the box must be clipped to the boundary."""
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    policy = ClampBounds(lower, upper)

    positions = np.array([[2.0, -3.0], [0.5, 0.5]])
    velocities = np.array([[1.0, 1.0], [0.1, 0.1]])
    new_pos, new_vel = policy.apply(positions, velocities)

    assert np.all(new_pos >= lower) and np.all(new_pos <= upper)
    assert np.array_equal(new_pos[1], positions[1])  # interior particle untouched


def test_clamp_bounds_zeroes_velocity_on_hit():
    """When a coordinate is clipped, the corresponding velocity must be zero."""
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    policy = ClampBounds(lower, upper)

    positions = np.array([[2.0, 0.5]])  # x0 exits, x1 stays inside
    velocities = np.array([[1.0, 0.3]])
    _, new_vel = policy.apply(positions, velocities)

    assert new_vel[0, 0] == 0.0   # x0 hit the wall -> velocity zeroed
    assert new_vel[0, 1] == 0.3   # x1 untouched
