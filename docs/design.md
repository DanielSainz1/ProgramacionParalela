# Design Document

## Architecture

The project is split into modules that each do one thing:

```
src/pso/
  core/           → PSO algorithm (doesn't know about parallelism)
  eval/           → The 5 evaluation strategies (V0–V4)
  objectives/     → Benchmark functions + SIR use case
  experiments/    → Config loading, orchestration and grid search
  io/             → Saving results to disk and metadata capture
  viz/            → Plots and animations
```

### How modules connect

```
scripts/*
  └── experiments/runner.py
        ├── experiments/config.py       (PSOConfig dataclass + YAML)
        ├── core/pso.py                 (run_pso, PSOResult)
        │     ├── core/state.py         (SwarmState)
        │     ├── core/bounds.py        (BoundsPolicy — injected)
        │     ├── core/topology.py      (Topology — injected)
        │     └── eval/base.py          (BaseEvaluator — injected)
        ├── eval/sequential.py          (V0)
        ├── eval/threading_eval.py      (V1)
        ├── eval/multiprocessing_eval.py (V2)
        ├── eval/async_eval.py          (V3)
        ├── eval/vectorized_eval.py     (V4)
        ├── objectives/*                (OBJECTIVES + OBJECTIVES_VEC registries)
        └── io/persistence.py           (save_run → JSON + CSV)
```

Dependencies go downward only — no circular imports.

Adding a new evaluator means creating one file in `eval/` and adding one line to the `EVALUATORS` dict in `runner.py`. Adding a new bounds policy or topology means implementing the ABC in `core/` — `run_pso` accepts both by dependency injection and never needs to change.

## Design decisions

### Strategy pattern for evaluators

`run_pso()` receives a `BaseEvaluator` object and calls `evaluate()`. It doesn't know or care whether it's sequential, threaded, multiprocessing, async or vectorised. All evaluators implement the same `open()`/`close()` lifecycle:

```python
class BaseEvaluator(ABC):
    def open(self) -> None: ...     # allocate resources (pool, event loop)
    def close(self) -> None: ...    # release resources
    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> np.ndarray: ...
```

The pool is created once in `open()` (called before the PSO loop) and destroyed in `close()` (in a `finally` block). An earlier version created/destroyed pools on every `evaluate()` call — that bug inflated V1 times by ~3x and V2 times by ~6x.

### Boundary handling

`BoundsPolicy` ABC with two implementations:

- **ClampBounds**: clips positions to `[lower, upper]` and zeroes velocity on hit axes. The particle stops at the wall and lets cognitive/social terms pull it back. Simple and stable.
- **ReflectBounds**: reflects positions off the boundary like a billiard ball and flips velocity. Conserves kinetic energy, better exploration near corners of the search space, at the cost of slightly less stable convergence near the optimum.

Both are tested and interchangeable via dependency injection.

### Topology

`Topology` ABC with two implementations:

- **GlobalBestTopology** (gbest): every particle is attracted to the single swarm-wide best. Fast convergence but prone to premature collapse on multi-modal functions.
- **RingTopology** (lbest): each particle only sees its `k` nearest neighbours in a logical ring. Slower convergence but much better diversity preservation — reduces the risk of getting trapped in local minima on Rastrigin/Ackley.

### Velocity clamping

Optional `vmax_ratio` caps `|v_i|` per dimension to `vmax_ratio × (upper - lower)`. Without it, particles accumulate extreme velocities and oscillate between boundaries without exploring the interior. Typical values are 0.2–0.5.

### Pickle validation (V2)

Instead of letting `ProcessPoolExecutor` fail with a cryptic error mid-run, `MultiprocessingEvaluator.open()` validates picklability upfront and raises a clear `TypeError` explaining why lambdas/closures can't cross process boundaries and what to do instead.

### on_iteration callback

Optional hook `on_iteration(it, state)` called after each iteration. Enables animation recording, custom convergence criteria, or live dashboards without modifying `run_pso`.

### JSON + CSV for persistence

JSON for config (hierarchical — params, timing, metadata including git hash and hardware info). CSV for per-iteration metrics (easy to load and plot). We considered SQLite but it's overkill for this project.

### YAML + CLI config

Default parameters in `configs/default.yaml`, CLI flags override individual values. Experiment quickly without editing files.

### Logging, not print

All modules use `logging.getLogger(__name__)`. Scripts configure the format. Library code never calls `basicConfig()` — Python best practice.

## Parallelism trade-offs

| | V0 Sequential | V1 Threading | V2 Multiprocessing | V3 Asyncio | V4 Vectorised |
|---|---|---|---|---|---|
| Overhead | None | Thread dispatch + GIL contention | Process creation + IPC + pickle | Event-loop scheduling | None (NumPy internal) |
| Real parallelism? | No | No (GIL) | Yes (separate processes) | No (single-threaded cooperative) | Implicit (BLAS/SIMD) |
| Best for | Cheap functions | I/O-bound work | Expensive CPU-bound functions (>1 ms) | I/O-bound with latency (APIs, sensors) | Vectorisable numerical objectives |
| Workers | N/A | Configurable | Configurable | N/A (single thread) | N/A (NumPy decides) |
| Batching | N/A | N/A | chunksize parameter | N/A | Full matrix at once |
| Pool lifecycle | N/A | open/close once | open/close once | Event loop open/close | N/A |

### When each variant wins

- **V4** wins on all cheap vectorisable fitnesses (4–15x over V0). No Python loop overhead, no GIL, no IPC — just contiguous memory and AVX/SSE.
- **V2** wins when per-particle fitness > ~1 ms and cannot be vectorised. Spreading 100 particles across 4 workers amortises the IPC cost. Confirmed on the SIR use case (2.19x speedup).
- **V3** wins when fitness includes real I/O latency (14–21x). `asyncio.gather` overlaps N waits so total time ≈ `max(latencies)` instead of `sum(latencies)`.
- **V1** never helps on CPU-bound Python code. The GIL serialises bytecode execution. Would help for I/O that releases the GIL.
- **V0** is the right baseline — for microsecond fitnesses, parallelism overhead exceeds the compute.

## Known limitations

- 4-core VM: results on a host with more physical cores could shift ratios slightly but not the ordering.
- Early stopping (tol=1e-10, stagnation=50) hurts us versus PySwarms, which runs all 500 iterations and keeps polishing.
- V4 only vectorises the evaluation step, not the particle update (which is also vectorisable — potential future work).
- The quality gap on Rastrigin/Rosenbrock d=30 vs PySwarms could be narrowed by combining RingTopology with vmax_ratio, but systematic tuning was out of scope.
- Animations only work for d=2 (contour + particles) and d=3 (scatter). Higher dimensions are not visualised.
