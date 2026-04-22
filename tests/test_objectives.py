"""Tests for the objective functions.

Verifies that each benchmark function returns the expected value at the
known global optimum and stays positive for non-optimal inputs.
"""
import numpy as np
import pytest

from pso.objectives.sphere import sphere
from pso.objectives.rosenbrock import rosenbrock
from pso.objectives.rastrigin import rastrigin
from pso.objectives.ackley import ackley


# -- Global optimum must be exactly zero ------------------------------------

@pytest.mark.parametrize("d", [2, 10, 30])
def test_sphere_zero_at_origin(d):
    assert sphere(np.zeros(d)) == 0.0


@pytest.mark.parametrize("d", [2, 10, 30])
def test_rosenbrock_zero_at_ones(d):
    assert rosenbrock(np.ones(d)) == 0.0


@pytest.mark.parametrize("d", [2, 10, 30])
def test_rastrigin_zero_at_origin(d):
    assert rastrigin(np.zeros(d)) == 0.0


@pytest.mark.parametrize("d", [2, 10, 30])
def test_ackley_zero_at_origin(d):
    assert ackley(np.zeros(d)) == pytest.approx(0.0, abs=1e-15)


# -- Positive away from optimum --------------------------------------------

def test_sphere_positive_away_from_origin():
    assert sphere(np.array([1.0, -2.0, 3.0])) > 0.0


def test_rosenbrock_positive_away_from_ones():
    assert rosenbrock(np.array([0.0, 0.0])) > 0.0


def test_rastrigin_positive_for_small_nonzero():
    """Rastrigin must stay positive for small non-zero inputs."""
    x = np.array([0.01, -0.01])
    value = rastrigin(x)
    assert value > 0.0


def test_ackley_positive_away_from_origin():
    assert ackley(np.array([1.0, 1.0])) > 0.0


# -- Known values -----------------------------------------------------------

def test_sphere_known_value():
    """sphere([1, 2, 3]) = 1 + 4 + 9 = 14."""
    assert sphere(np.array([1.0, 2.0, 3.0])) == pytest.approx(14.0)
