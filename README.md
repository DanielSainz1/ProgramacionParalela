# Particle Swarm Optimization -- Parallel Programming

Particle Swarm Optimization (PSO) in Python with three interchangeable
evaluators (sequential, threading, multiprocessing) built on the Strategy
pattern. Includes multi-seed timing experiments, a V2 batching study, and a
PySwarms baseline. The full write-up with real numbers and analysis lives in
[`docs/report.md`](docs/report.md).

**TL;DR of findings (see report for details):**

- Sequential (V0) is fastest on every benchmark — cheap fitness functions
  make parallelism a net loss.
- Threading (V1) is ~10x slower than V0 due to the GIL.
- Multiprocessing (V2) is ~30x slower than V0 at `chunksize=1`; batching at
  `chunksize=64` recovers ~4x but never crosses V0.
- We beat PySwarms on smooth high-dim problems (Sphere d=30, Ackley d=30) and
  lose on multimodal ones (Rastrigin, Rosenbrock).

---

## Installation

```bash
git clone <repository-url>
cd ProgramacionParalela
python -m venv zonaproyecto
source zonaproyecto/bin/activate
pip install -e ".[dev]"
```

Dependencies: NumPy, Matplotlib, PyYAML (installed automatically).

---

## Commands

| Command | Description |
|---|---|
| `python scripts/run_pso.py` | Single PSO run (default: sphere, d=30, seed=42) |
| `python scripts/run_pso.py --objective rastrigin --dim 10 --seed 99` | Custom parameters |
| `python scripts/run_pso.py --evaluator threading` | Choose evaluator (sequential, threading, multiprocessing) |
| `python scripts/run_pso.py --profile` | Profile execution with cProfile |
| `python scripts/run_pso.py --config configs/default.yaml` | Load custom config |
| `python scripts/run_benchmarks.py` | Full benchmark: 4 functions x 3 dims x 3 evaluators = 36 runs |
| `python scripts/run_grid_search.py --objective sphere --dim 2` | Grid search over w, c1, c2 |
| `python scripts/run_comparison.py` | Multi-seed speedup comparison V0 vs V1 vs V2 → `results/comparison.csv` |
| `python scripts/run_batching_experiment.py` | V2 `chunksize` sweep (1..128) → `results/batching.csv` + `.png` |
| `python scripts/run_pyswarms_baseline.py` | Convergence vs PySwarms library → `results/pyswarms_baseline.csv` |
| `python scripts/make_viz.py --run-dir results/<folder>/` | Generate plots and animations |
| `python scripts/make_viz.py --run-dir results/<folder>/ --type convergence` | Only convergence plot |
| `python scripts/analyze_results.py --results-dir results/` | Convergence comparison, boxplot, summary table |
| `python scripts/analyze_results.py --evaluator threading` | Filter analysis by evaluator |
| `python scripts/run_evaluator_demo.py` | Quick demo of SequentialEvaluator |
| `pytest` | Run unit tests |

---

## Architecture

```
src/pso/
├── core/               # PSO engine
│   ├── pso.py          # run_pso() — main loop, returns PSOResult
│   ├── state.py        # SwarmState dataclass (positions, velocities, pbest, gbest)
│   ├── bounds.py       # BoundsPolicy ABC + ClampBounds (also free clamp_positions)
│   └── topology.py     # Topology ABC + GlobalBestTopology
│
├── eval/               # Fitness evaluators (strategy pattern)
│   ├── base.py         # BaseEvaluator — abstract base class
│   ├── sequential.py   # V0: SequentialEvaluator — baseline loop
│   ├── threading_eval.py    # V1: ThreadingEvaluator — ThreadPoolExecutor
│   └── multiprocessing_eval.py  # V2: MultiprocessingEvaluator — ProcessPoolExecutor
│
├── objectives/         # Benchmark functions
│   ├── sphere.py       # f(x) = sum(x^2)
│   ├── rosenbrock.py   # Curved valley
│   ├── rastrigin.py    # Many local minima
│   └── ackley.py       # Flat deceptive region
│
├── experiments/        # Orchestration
│   ├── config.py       # PSOConfig dataclass + from_yaml()
│   ├── runner.py       # run_pso_from_config() + EVALUATORS registry
│   └── grid_search.py  # grid_search() — parameter sweep
│
├── io/                 # Persistence
│   ├── metadata.py     # get_git_hash(), get_hardware_info()
│   └── persistence.py  # save_run() — config.json + metrics.csv
│
└── viz/                # Visualization
    ├── convergence.py      # plot_convergence() — best fitness vs iteration
    ├── swarm_animation.py  # animate_swarm_2d() — particle movement GIF
    └── swarm_3d.py         # animate_swarm_3d() — 3D particle movement GIF

scripts/
├── run_pso.py                   # Single run CLI (argparse)
├── run_benchmarks.py            # Full benchmark suite (4 obj x 3 dims x 3 evaluators)
├── run_grid_search.py           # Hyperparameter grid search
├── run_comparison.py            # Multi-seed speedup V0 vs V1 vs V2
├── run_batching_experiment.py   # V2 chunksize sweep
├── run_pyswarms_baseline.py     # Convergence vs PySwarms library
├── make_viz.py                  # Convergence plots + swarm animations
├── analyze_results.py           # Convergence comparison, boxplots, summary table
└── run_evaluator_demo.py        # Quick demo of evaluator usage

configs/
└── default.yaml        # Default PSO parameters

tests/
├── test_reproducibility.py         # Same seed = same result
├── test_bounds.py                  # Particles stay within bounds
├── test_bounds_policy.py           # ClampBounds clips and zeroes velocity on wall hit
├── test_topology.py                # GlobalBestTopology broadcasts gbest correctly
├── test_monotonic_gbest.py         # gbest never worsens
├── test_sphere_convergence.py      # Converges to ~0 on sphere
├── test_evaluator_equivalence.py   # V0 / V1 / V2 give equivalent results
└── test_grid_search.py             # Grid search writes a valid CSV
```

### Module dependencies

```
scripts/*
  └── experiments/runner.py      (run_pso_from_config)
        ├── experiments/config.py  (PSOConfig, from_yaml)
        ├── core/pso.py            (run_pso, PSOResult)
        │     ├── core/state.py    (SwarmState)
        │     ├── core/bounds.py   (clamp_positions)
        │     └── eval/base.py     (BaseEvaluator — injected)
        ├── eval/sequential.py     (V0)
        ├── eval/threading_eval.py (V1)
        ├── eval/multiprocessing_eval.py (V2)
        ├── objectives/*           (OBJECTIVES registry)
        └── io/persistence.py      (save_run)
              └── io/metadata.py   (git hash, hardware info)
```

---

## Architecture patterns

`run_pso()` is agnostic of the evaluator, the boundary handling, and the
neighbourhood structure. All three are injected via ABCs (Strategy pattern),
so new implementations can be added without touching the optimisation loop.

```
BaseEvaluator (ABC)                 BoundsPolicy (ABC)       Topology (ABC)
├── SequentialEvaluator   (V0)      └── ClampBounds          └── GlobalBestTopology
├── ThreadingEvaluator    (V1)
└── MultiprocessingEvaluator (V2)
```

All evaluators implement the same interface:

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        ...
```

`BoundsPolicy.apply(positions, velocities)` enforces box constraints and can
modify velocities (e.g. zero them on the axis that hit the wall).
`Topology.social_best_positions(pbest, pbest_costs)` returns each particle's
social best — currently the global best broadcast to all, but a ring or
von-Neumann topology would plug in here.

To switch evaluator, use the `--evaluator` flag or change `evaluator:` in the
YAML config.

### V0 -- Sequential (baseline)

Evaluates each particle one by one in a simple loop. No parallelism overhead. This is the fastest option for cheap objective functions.

### V1 -- Threading (ThreadPoolExecutor)

Uses `concurrent.futures.ThreadPoolExecutor` with `executor.map()` to evaluate particles in parallel across multiple threads.

**Limitation**: Python's GIL (Global Interpreter Lock) prevents true parallelism for CPU-bound code. Since our objective functions are pure Python/NumPy computations, threading does **not** yield speedup here. Threading is beneficial for I/O-bound workloads (network calls, file reads).

### V2 -- Multiprocessing (ProcessPoolExecutor)

Uses `concurrent.futures.ProcessPoolExecutor` with `executor.map()` and `chunksize` parameter to evaluate particles across separate OS processes.

**Advantage**: Each process has its own Python interpreter and GIL, so CPU-bound work can run in true parallel on multiple cores.

**Overhead**: Processes require serialization (pickling) of data via IPC (inter-process communication). For cheap functions like sphere/rastrigin, this overhead dominates. The `chunksize` parameter (default=10) reduces IPC by batching multiple particles per task instead of sending them one by one.

**When it helps**: When the objective function is computationally expensive (e.g., simulations, ML model evaluation), the parallel computation outweighs the IPC cost and produces real speedup.

### Speedup comparison

Run `python scripts/run_comparison.py` to measure V0 vs V1 vs V2 across all 4 functions and 3 dimensions (2, 10, 30). Uses `time.perf_counter` for precise timing.

For our benchmark functions (pure NumPy, microsecond evaluation), sequential is fastest because the overhead of thread/process management exceeds the computation time. This is expected and documented — parallelism pays off only when evaluation cost is significant.

---

## Benchmark functions

| Function | Global minimum | Bounds | Difficulty |
|---|---|---|---|
| Sphere | f(0,...,0) = 0 | [-100, 100] | Low -- simple unimodal |
| Rosenbrock | f(1,...,1) = 0 | [-5, 10] | Medium -- curved valley |
| Rastrigin | f(0,...,0) = 0 | [-5.12, 5.12] | High -- many local minima |
| Ackley | f(0,...,0) = 0 | [-32.768, 32.768] | High -- flat deceptive region |

Benchmarks run at d=2, d=10, d=30 with reproducible seeds.

---

## PSO parameters (default.yaml)

| Parameter | Value | Description |
|---|---|---|
| `w` | 0.719 | Inertia weight (Clerc-Kennedy constriction) |
| `c1` | 1.49445 | Cognitive coefficient (personal best attraction) |
| `c2` | 1.49445 | Social coefficient (global best attraction) |
| `n_particles` | 100 | Swarm size |
| `max_iter` | 500 | Maximum iterations |
| `seed` | 42 | Random seed for reproducibility |

---

## Design decisions

- **Boundary handling**: `BoundsPolicy` ABC with `ClampBounds` as the default. It clips positions to the box **and** zeroes velocities on any axis that hit a wall, so particles don't keep bouncing off the boundary.
- **Topology**: `Topology` ABC with `GlobalBestTopology` as the default (all particles see the swarm-wide best). A ring or lbest topology can be added by implementing `social_best_positions`.
- **Evaluator injection**: `BaseEvaluator` ABC. The PSO core receives an evaluator object, making it trivial to swap parallelism strategies without modifying the algorithm.
- **Configuration**: YAML file + CLI overrides via argparse. YAML provides defaults; CLI flags override individual parameters.
- **Persistence**: JSON for config (structured, includes git hash and hardware info), CSV for metrics (one row per iteration, easy to load with pandas/numpy).
- **Logging**: Structured logging via Python's `logging` module. Shows timestamp, level, iteration, and best fitness. Configured in scripts via `logging.basicConfig()`.
- **Results directory**: `results/` is gitignored to keep the repo clean. Each run saves to a timestamped subfolder.

---

## Saved results

Each run saves to `results/<timestamp>_<objective>_d<dim>_s<seed>/`:

- `config.json` -- full parameters + git commit hash + hardware info
- `metrics.csv` -- per-iteration best fitness history

---

## Tests

```bash
pytest
```

| Test | What it verifies |
|---|---|
| `test_reproducibility` | Same seed produces identical results |
| `test_bounds` | Particles never escape search bounds |
| `test_bounds_policy` (2) | `ClampBounds` clips positions and zeroes velocity on wall hits |
| `test_topology` (2) | `GlobalBestTopology` broadcasts gbest and returns independent copies |
| `test_monotonic_gbest` | Global best fitness never worsens across iterations |
| `test_sphere_convergence` | Converges to ~0 on sphere with reasonable parameters |
| `test_v0_v1_same_result` | V0 and V1 give exactly the same result (threading preserves order) |
| `test_v0_v2_comparable_result` | V0 and V2 give equivalent results (tolerance 1e-12 for IPC) |
| `test_grid_search_generates_csv` | Grid search produces a valid CSV with correct columns |

**Total: 11 tests, all passing.**

---

## Reproducibility

- All runs accept a `seed` parameter (NumPy `default_rng`)
- Config is saved alongside results (exact parameters + git hash)
- Hardware info is recorded for cross-machine comparison
- Timing uses `time.perf_counter` for precision
