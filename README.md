# Particle Swarm Optimization -- Parallel Programming

Particle Swarm Optimization (PSO) in Python with five interchangeable
evaluators (sequential, threading, multiprocessing, async, vectorized) built
on the Strategy pattern. Includes multi-seed timing experiments, a V2 batching
study, a V3 latency experiment, a PySwarms baseline, and a real-world use case
calibrating an SIR epidemic model from noisy daily infection counts. The full
write-up with real numbers and analysis lives in [`docs/report.md`](docs/report.md).

**TL;DR of findings (see report for details):**

- **V4 (NumPy vectorised)** is the fastest of all variants: 7-55x speedup vs V0
  on the benchmark objectives. It eliminates the Python interpreter overhead
  by evaluating all particles at once via BLAS/SIMD.
- **V0 (sequential)** is the baseline; for cheap objectives, classic
  parallelism (V1/V2) costs more than it saves.
- **V1 (threading)** is ~3–7x slower than V0 due to the GIL.
- **V2 (multiprocessing)** is ~5–17x slower than V0; batching at `chunksize=64`
  gives 13x improvement over chunksize=1 but never crosses V0.
- **V3 (asyncio)** matches V0 with zero latency (event-loop overhead is tiny),
  but wins dramatically — by orders of magnitude — when each evaluation has
  I/O-style latency. See `scripts/run_latency_experiment.py`.
- **SIR use case**: on a real fitness costing ~1 ms per particle, V2
  multiprocessing **finally wins** (2.19x speedup) and V4 still wins (1.89x).
  Both correctly recover the ground-truth parameters (β=0.30, γ=0.10, I₀=10)
  from a noisy synthetic curve. This flips the cheap-benchmark verdict.
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
| `python scripts/run_pso.py --evaluator vectorized` | Choose evaluator (sequential, threading, multiprocessing, async, vectorized) |
| `python scripts/run_pso.py --profile` | Profile execution with cProfile |
| `python scripts/run_pso.py --config configs/default.yaml` | Load custom config |
| `python scripts/run_benchmarks.py` | Full benchmark: 4 functions x 3 dims x 3 evaluators = 36 runs |
| `python scripts/run_grid_search.py --objective sphere --dim 2` | Grid search over w, c1, c2 |
| `python scripts/run_comparison.py` | Multi-seed speedup comparison V0 vs V1 vs V2 vs V3 vs V4 |
| `python scripts/run_batching_experiment.py` | V2 `chunksize` sweep (1..128) |
| `python scripts/run_latency_experiment.py` | V0 vs V3 with simulated I/O latency |
| `python scripts/generate_sir_observations.py` | Generate the synthetic SIR ground-truth CSV (run once) |
| `python scripts/run_sir_comparison.py` | All 5 variants on the SIR calibration use case |
| `python scripts/run_pyswarms_baseline.py` | Convergence vs PySwarms library |
| `python scripts/make_viz.py --run-dir results/<folder>/` | Generate plots and animations |
| `python scripts/analyze_results.py --results-dir results/` | Convergence comparison, boxplot, summary table |
| `pytest` | Run unit tests (76 tests) |

---

## Architecture

```
src/pso/
├── core/               # PSO engine
│   ├── pso.py          # run_pso() — main loop, returns PSOResult
│   ├── state.py        # SwarmState dataclass (positions, velocities, pbest, gbest)
│   ├── bounds.py       # BoundsPolicy ABC + ClampBounds, ReflectBounds
│   └── topology.py     # Topology ABC + GlobalBestTopology, RingTopology
│
├── eval/               # Fitness evaluators (strategy pattern)
│   ├── base.py         # BaseEvaluator ABC (open/close lifecycle)
│   ├── sequential.py            # V0: baseline loop
│   ├── threading_eval.py        # V1: ThreadPoolExecutor
│   ├── multiprocessing_eval.py  # V2: ProcessPoolExecutor + batching
│   ├── async_eval.py            # V3: asyncio.gather with simulated latency
│   └── vectorized_eval.py       # V4: NumPy BLAS / SIMD on the full matrix
│
├── objectives/         # Benchmark functions (scalar + vectorised pair)
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

tests/                  # 73 tests across 16 files
├── test_objectives.py          # f(optimum)=0, positivity, known values
├── test_sphere_convergence.py  # Convergence at d=2,10 across seeds
├── test_monotonic_gbest.py     # gbest never worsens
├── test_bounds.py              # Particles stay within bounds
├── test_bounds_policy.py       # ClampBounds + ReflectBounds correctness
├── test_topology.py            # GlobalBest + Ring correctness
├── test_reproducibility.py     # Same seed = same result
├── test_pool_lifecycle.py      # open/close/reuse, pickle validation
├── test_vmax.py                # Velocity clamping
├── test_on_iteration.py        # Callback hook
├── test_evaluator_equivalence.py  # V0/V1/V2 give same results
├── test_persistence.py         # save_run creates correct files
└── test_grid_search.py         # Grid search CSV output
```

---

## Architecture patterns

`run_pso()` is agnostic of the evaluator, the boundary handling, and the
neighbourhood structure. All three are injected via ABCs (Strategy pattern),
so new implementations can be added without touching the optimisation loop.

```
BaseEvaluator (ABC)                    BoundsPolicy (ABC)       Topology (ABC)
├── SequentialEvaluator      (V0)      ├── ClampBounds          ├── GlobalBestTopology
├── ThreadingEvaluator       (V1)      └── ReflectBounds        └── RingTopology
├── MultiprocessingEvaluator (V2)
├── AsyncEvaluator           (V3)
└── VectorizedEvaluator      (V4)
```

All evaluators implement the same interface with an `open()`/`close()` lifecycle:

```python
class BaseEvaluator(ABC):
    def open(self) -> None: ...   # allocate resources (pool)
    def close(self) -> None: ...  # release resources
    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> np.ndarray: ...
```

`BoundsPolicy.apply(positions, velocities)` enforces box constraints and can
modify velocities (ClampBounds zeroes them, ReflectBounds flips them).

`Topology.social_best_positions(pbest, pbest_costs, gbest)` returns each
particle's social best — GlobalBest broadcasts gbest to all particles, Ring
restricts it to the k nearest neighbours.

### V0 -- Sequential (baseline)

Evaluates each particle one by one in a simple loop. No parallelism overhead.
This is the fastest option for cheap objective functions.

### V1 -- Threading (ThreadPoolExecutor)

Uses `concurrent.futures.ThreadPoolExecutor` to evaluate particles in parallel
across multiple threads. Pool is created once in `open()` and reused.

**Limitation**: Python's GIL prevents true parallelism for CPU-bound code.
Threading is beneficial for I/O-bound workloads (network calls, file reads).

### V2 -- Multiprocessing (ProcessPoolExecutor)

Uses `concurrent.futures.ProcessPoolExecutor` with batch splitting to evaluate
particles across separate OS processes. Pool is created once in `open()`.
Validates picklability of the objective function upfront.

**Advantage**: Each process has its own GIL, so CPU-bound work runs truly
parallel on multiple cores.

**Overhead**: IPC (pickling + pipe transfer) dominates for cheap functions.
The `chunksize` parameter reduces IPC by batching particles per task.

### V3 -- Asyncio (cooperative concurrency)

Wraps each particle's evaluation in a coroutine that first awaits a
configurable simulated latency (`latency_ms_min`..`latency_ms_max`) and then
computes the objective. All N coroutines are launched with `asyncio.gather`,
so their sleeps overlap on a single thread — the event loop hands control to
the next coroutine while one is waiting.

**When it helps**: I/O-bound fitnesses — remote sensor calls, database
queries, queued microservices. `run_latency_experiment.py` shows V3 winning
by orders of magnitude once each evaluation includes real latency.

**When it does not**: pure CPU work with zero latency. asyncio adds event-loop
dispatch overhead and gives nothing back, since a single thread cannot
parallelise compute.

### V4 -- NumPy vectorised (implicit parallelism)

Replaces the per-particle Python loop with a single matrix operation. Each
objective in `objectives/` ships a scalar version `f(x: (d,)) -> float` and
a vectorised version `f_vec(X: (N, d)) -> (N,)` registered in
`OBJECTIVES_VEC`. The evaluator just calls `f_vec(positions)` and lets
NumPy/BLAS handle the dispatch to SIMD instructions.

**Why it wins**: no Python interpreter overhead, no GIL, no IPC, no event
loop. Just contiguous memory and AVX/SSE. Measured speedups vs V0 range
from 7x (Rastrigin) to 55x (Sphere) on the benchmark suite.

---

## Benchmark functions

| Function | Global minimum | Bounds | Difficulty |
|---|---|---|---|
| Sphere | f(0,...,0) = 0 | [-100, 100] | Low -- simple unimodal |
| Rosenbrock | f(1,...,1) = 0 | [-5, 10] | Medium -- curved valley |
| Rastrigin | f(0,...,0) = 0 | [-5.12, 5.12] | High -- many local minima |
| Ackley | f(0,...,0) = 0 | [-32.768, 32.768] | High -- flat deceptive region |

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
| `vmax_ratio` | None | Velocity clamp as fraction of search range (e.g. 0.5) |

---

## Design decisions

- **Boundary handling**: `BoundsPolicy` ABC with two implementations. `ClampBounds` clips positions and zeroes velocity on wall hits. `ReflectBounds` mirrors excess distance and flips velocity, conserving kinetic energy.
- **Topology**: `Topology` ABC with two implementations. `GlobalBestTopology` (fast convergence) and `RingTopology` (better diversity on multimodal functions).
- **Velocity clamping**: Optional `vmax_ratio` caps particle speed to a fraction of the search range, preventing overshoot oscillations.
- **Evaluator lifecycle**: `open()`/`close()` pattern creates the thread/process pool once per PSO run instead of per-evaluate call. This eliminated a ~3–6x overhead bug.
- **Pickle validation**: `MultiprocessingEvaluator.open()` checks that the objective can be serialized before creating the pool, with a clear error message.
- **on_iteration callback**: Optional hook for custom per-iteration behaviour (animation, logging, early stopping) without modifying the core loop.
- **Configuration**: YAML file + CLI overrides via argparse.
- **Persistence**: JSON for config (includes git hash and hardware info), CSV for per-iteration metrics.

---

## Tests

```bash
pytest
```

**76 tests across 16 files**, covering: objective function correctness,
convergence, monotonic gbest, bounds enforcement, both bounds policies, both
topologies, reproducibility, pool lifecycle, pickle validation, velocity
clamping, on_iteration callback, evaluator equivalence, persistence, and grid
search.

---

## Reproducibility

- All runs accept a `seed` parameter (NumPy `default_rng`)
- Config is saved alongside results (exact parameters + git hash)
- Hardware info is recorded for cross-machine comparison
- Timing uses `time.perf_counter` for precision
