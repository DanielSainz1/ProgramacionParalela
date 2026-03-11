# Particle Swarm Optimization — Parallel Programming

A complete implementation of Particle Swarm Optimization (PSO) in Python for a university parallel programming course. The project benchmarks sequential and parallel evaluation strategies across standard optimization functions.

---

## Installation

```bash
git clone <repository-url>
cd ProgramacionParalela
python -m venv zonaproyecto
source zonaproyecto/bin/activate
pip install -e ".[dev]"
```

---

## Usage

### Single PSO run

```bash
# Default settings (sphere, d=30, seed=42)
python scripts/run_pso.py

# Custom parameters via CLI
python scripts/run_pso.py --objective rastrigin --dim 10 --seed 99

# Profile execution time (shows which functions are slowest)
python scripts/run_pso.py --objective sphere --dim 10 --profile

# Load a custom config file
python scripts/run_pso.py --config configs/default.yaml
```

### Full benchmark suite

```bash
python scripts/run_benchmarks.py
```

Runs all 4 objective functions × 3 dimensions = 12 experiments and saves results to `results/`.

### Hyperparameter grid search

```bash
python scripts/run_grid_search.py --objective sphere --dim 2
```

Tries combinations of `w`, `c1`, `c2` across multiple seeds and saves a summary to `results/grid_search.csv`.

### Generate visualizations

```bash
python scripts/make_viz.py --run-dir results/<run-folder>/
```

Generates:
- `convergence.png` — best fitness vs iteration curve
- `swarm.gif` — swarm animation (only when `dim=2`)

### Run tests

```bash
pytest
```

---

## Project structure

```
src/pso/
├── core/           # PSO engine: run_pso(), SwarmState, boundary handling
├── eval/           # Evaluators: BaseEvaluator, SequentialEvaluator
├── objectives/     # Benchmark functions: sphere, rosenbrock, rastrigin, ackley
├── experiments/    # Config, runner, grid search
├── io/             # Persistence: save_run(), hardware metadata
└── viz/            # Visualization: convergence plots, swarm animation

scripts/
├── run_pso.py          # Single run with CLI
├── run_benchmarks.py   # Full benchmark suite
├── run_grid_search.py  # Hyperparameter grid search
└── make_viz.py         # Generate plots and animations

configs/
└── default.yaml        # Default PSO configuration

tests/                  # Unit tests (pytest)
results/                # Saved experiment outputs (gitignored)
```

---

## Benchmark functions

| Function | Global minimum | Bounds | Difficulty |
|---|---|---|---|
| Sphere | f(0,...,0) = 0 | [-100, 100] | Low — simple unimodal |
| Rosenbrock | f(1,...,1) = 0 | [-5, 10] | Medium — curved valley |
| Rastrigin | f(0,...,0) = 0 | [-5.12, 5.12] | High — many local minima |
| Ackley | f(0,...,0) = 0 | [-32.768, 32.768] | High — flat deceptive region |

---

## PSO parameters (default.yaml)

| Parameter | Value | Description |
|---|---|---|
| `w` | 0.719 | Inertia weight (Clerc-Kennedy) |
| `c1` | 1.49445 | Cognitive coefficient |
| `c2` | 1.49445 | Social coefficient |
| `n_particles` | 100 | Swarm size |
| `max_iter` | 500 | Maximum iterations |

---

## Saved results

Each run saves to `results/<timestamp>_<objective>_d<dim>_s<seed>/`:

- `config.json` — full parameters + git commit hash + hardware info
- `metrics.csv` — per-iteration convergence history

---

## Parallelism strategy

The core PSO algorithm stays identical across all variants. Only the evaluator changes:

```
BaseEvaluator (abstract)
├── SequentialEvaluator   ← V0: baseline
├── ThreadingEvaluator    ← V1: ThreadPoolExecutor
├── ProcessEvaluator      ← V2: ProcessPoolExecutor
├── AsyncEvaluator        ← V3: asyncio
└── VectorizedEvaluator   ← V4: NumPy vectorized
```

To switch evaluator, change one line in the config:
```yaml
evaluator: sequential   # or threading, multiprocessing, asyncio, vectorized
```

| Version | Strategy | Status |
|---|---|---|
| V0 | Sequential (baseline) | ✅ |
| V1 | Threading | ⬜ |
| V2 | Multiprocessing | ⬜ |
| V3 | Asyncio | ⬜ |
| V4 | NumPy vectorization | ⬜ |
