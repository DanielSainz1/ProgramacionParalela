import numpy as np
from pso.core.topology import GlobalBestTopology, RingTopology
from pso.core.pso import run_pso
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator


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


def test_ring_topology_picks_local_best():
    """Each particle must get the best pbest from its neighbourhood."""
    topo = RingTopology(k=1)
    # 4 particles, d=1 for simplicity
    pbest_positions = np.array([[10.0], [20.0], [30.0], [40.0]])
    pbest_values = np.array([5.0, 1.0, 8.0, 3.0])  # particle 1 is best overall
    gbest = pbest_positions[1]

    social = topo.social_best_positions(pbest_positions, pbest_values, gbest)
    # Particle 0 neighbours: {3, 0, 1} -> best is 1 (val=1.0)
    assert social[0, 0] == 20.0
    # Particle 2 neighbours: {1, 2, 3} -> best is 1 (val=1.0)
    assert social[2, 0] == 20.0
    # Particle 3 neighbours: {2, 3, 0} -> best is 3 (val=3.0)
    assert social[3, 0] == 40.0


def test_ring_topology_pso_converges():
    """PSO with RingTopology must still converge on sphere."""
    d = 2
    lower = np.full(d, -10.0)
    upper = np.full(d, 10.0)
    ev = SequentialEvaluator(sphere)
    result = run_pso(sphere, d, 30, 400, 0.719, 1.49445, 1.49445,
                     lower, upper, ev, seed=42,
                     topology=RingTopology(k=1))
    assert result.best_value < 1e-4
