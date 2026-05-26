import numpy as np
import pytest
from pso.core.bounds import ClampBounds, ReflectBounds
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator


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


def test_reflect_bounds_stays_in_box():
    """Reflected positions must always land inside [lower, upper]."""
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    policy = ReflectBounds(lower, upper)

    positions = np.array([[2.5, -3.0], [0.5, 0.5], [1.3, -1.7]])
    velocities = np.array([[1.0, -1.0], [0.1, 0.1], [0.5, -0.5]])
    new_pos, new_vel = policy.apply(positions, velocities)

    assert np.all(new_pos >= lower) and np.all(new_pos <= upper)
    assert np.allclose(new_pos[1], positions[1])  # interior particle untouched


def test_reflect_bounds_flips_velocity():
    """Velocity must be negated on axes that crossed the boundary."""
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    policy = ReflectBounds(lower, upper)

    positions = np.array([[2.0, 0.5]])   # x0 exits above, x1 inside
    velocities = np.array([[1.0, 0.3]])
    _, new_vel = policy.apply(positions, velocities)

    assert new_vel[0, 0] == -1.0  # flipped
    assert new_vel[0, 1] == 0.3   # untouched


def test_reflect_bounds_pso_converges():
    """PSO with ReflectBounds must still converge on sphere."""
    d = 2
    lower = np.full(d, -10.0)
    upper = np.full(d, 10.0)
    ev = SequentialEvaluator(sphere)
    result = run_pso(sphere, d, 30, 300, 0.719, 1.49445, 1.49445,
                     lower, upper, ev, seed=42,
                     bounds_policy=ReflectBounds(lower, upper))
    assert result.best_value < 1e-5


def test_reflect_bounds_pso_converges_d10():
    """ReflectBounds must also work on higher dimensions."""
    d = 10
    lower = np.full(d, -10.0)
    upper = np.full(d, 10.0)
    ev = SequentialEvaluator(sphere)
    result = run_pso(sphere, d, 60, 500, 0.719, 1.49445, 1.49445,
                     lower, upper, ev, seed=42,
                     bounds_policy=ReflectBounds(lower, upper))
    assert result.best_value < 1e-3
