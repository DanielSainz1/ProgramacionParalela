"""Tests for the IO/persistence layer.

Verifies that save_run produces valid JSON config and CSV metrics files.
"""
import json
import csv
from pathlib import Path

import numpy as np

from pso.core.pso import run_pso, PSOResult
from pso.experiments.config import PSOConfig
from pso.io.persistence import save_run
from pso.objectives.sphere import sphere
from pso.eval.sequential import SequentialEvaluator


def _quick_result() -> tuple[PSOConfig, PSOResult]:
    """Run a tiny PSO to get a real PSOResult for persistence tests."""
    cfg = PSOConfig(
        objective="sphere", dim=2, n_particles=10, max_iter=20,
        w=0.7, c1=1.5, c2=1.5, lower=-5.0, upper=5.0, seed=42,
    )
    lower = np.full(cfg.dim, cfg.lower)
    upper = np.full(cfg.dim, cfg.upper)
    result = run_pso(
        sphere, cfg.dim, cfg.n_particles, cfg.max_iter,
        cfg.w, cfg.c1, cfg.c2, lower, upper,
        SequentialEvaluator(sphere), seed=cfg.seed,
    )
    return cfg, result


def test_save_run_creates_json_and_csv(tmp_path):
    cfg, result = _quick_result()
    run_dir = save_run(cfg, result, out_dir=str(tmp_path))

    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.csv").exists()


def test_save_run_json_has_expected_fields(tmp_path):
    cfg, result = _quick_result()
    run_dir = save_run(cfg, result, out_dir=str(tmp_path))

    with open(run_dir / "config.json") as f:
        data = json.load(f)

    # Top-level sections
    assert "pso" in data
    assert "timing" in data
    assert "result" in data
    assert "meta" in data

    # PSO config fields
    assert data["pso"]["objective"] == "sphere"
    assert data["pso"]["dim"] == 2
    assert data["pso"]["seed"] == 42

    # Timing fields
    assert "total_time" in data["timing"]
    assert "eval_time" in data["timing"]

    # Result fields
    assert "best_value" in data["result"]
    assert isinstance(data["result"]["best_position"], list)

    # Metadata
    assert "git_hash" in data["meta"]
    assert "hardware" in data["meta"]


def test_save_run_csv_has_correct_columns(tmp_path):
    cfg, result = _quick_result()
    run_dir = save_run(cfg, result, out_dir=str(tmp_path))

    with open(run_dir / "metrics.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == len(result.best_history)
    assert "iteration" in rows[0]
    assert "gbest_value" in rows[0]
    # Values should be parseable as floats
    assert float(rows[0]["gbest_value"]) >= 0.0
