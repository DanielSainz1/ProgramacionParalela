import numpy as np
from pso.core.topology import GlobalBestTopology


def test_global_best_broadcasts_gbest_to_all_particles():
    """GlobalBestTopology must return gbest_position for every particle."""
    topology = GlobalBestTopology()
    n_particles, d = 4, 3
    pbest_positions = np.random.rand(n_particles, d)
    pbest_values = np.random.rand(n_particles)
    gbest = np.array([0.5, 0.5, 0.5])

    social = topology.social_best_positions(pbest_positions, pbest_values, gbest)
    assert social.shape == (n_particles, d)
    for i in range(n_particles):
        assert np.array_equal(social[i], gbest)


def test_global_best_returns_independent_copy():
    """Modifying the returned array must not modify the original gbest."""
    topology = GlobalBestTopology()
    gbest = np.array([1.0, 2.0, 3.0])
    pbest_positions = np.zeros((3, 3))
    pbest_values = np.zeros(3)

    social = topology.social_best_positions(pbest_positions, pbest_values, gbest)
    social[0, 0] = 999.0
    assert gbest[0] == 1.0  # original unchanged
