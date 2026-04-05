import csv
import os
from pso.experiments.config import PSOConfig
from pso.experiments.grid_search import grid_search

EXPECTED_COLUMNS = [
    "evaluator", "n_particles", "max_iter", "w", "c1", "c2", "seed",
    "best_value", "total_time", "eval_time", "update_time", "overhead",
    "auc", "convergence_iter",
]


def test_grid_search_generates_csv(tmp_path):
    """grid_search() must create a CSV with correct columns and row count."""
    cfg = PSOConfig(
        objective="sphere", dim=2, n_particles=10, max_iter=50,
        w=0.7, c1=1.5, c2=1.5, lower=-10.0, upper=10.0, seed=42,
    )
    w_values = [0.4, 0.7]
    c1_values = [1.5]
    c2_values = [1.5]
    seeds = [42, 99]

    out_path = str(tmp_path / "grid.csv")
    result_path = grid_search(cfg, w_values, c1_values, c2_values, seeds, out_path=out_path)

    # File exists
    assert os.path.isfile(result_path)

    # Read and validate
    with open(result_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Correct columns
    assert reader.fieldnames == EXPECTED_COLUMNS

    # Expected rows: 2 w_values * 1 c1 * 1 c2 * 2 seeds = 4
    assert len(rows) == 4

    # Every row has a numeric best_value
    for row in rows:
        assert float(row["best_value"]) >= 0.0
