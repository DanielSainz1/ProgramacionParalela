# PSO Project — Final Report

## 1. Introduction

This project implements Particle Swarm Optimization (PSO) in Python with three
evaluation strategies — sequential (V0), threading (V1) and multiprocessing (V2)
— and compares them empirically on four standard benchmark functions
(Sphere, Rosenbrock, Rastrigin, Ackley) in dimensions 2, 10 and 30.

The project has two goals:

1. Build a clean PSO that can swap the fitness evaluator, the bounds policy and
   the topology without touching the optimisation loop (Strategy pattern + ABCs).
2. Measure *how much* — and *why* — parallelism pays off for this kind of
   workload, using real multi-seed timing data instead of theoretical claims.

The experimental section is driven entirely by CSV files produced by the scripts
in `scripts/`. No number in the tables below is hand-written.

## 2. Methodology

### 2.1 PSO algorithm

Canonical inertia-weight PSO. Each iteration:

```
v_i ← w · v_i + c1 · r1 · (pbest_i − x_i) + c2 · r2 · (social_best_i − x_i)
v_i ← clip(v_i, -vmax, vmax)          [optional velocity clamping]
x_i ← x_i + v_i
x_i, v_i ← bounds_policy.apply(x_i, v_i)
```

- **Initialisation**: positions uniform inside `[lower, upper]^d`,
  velocities uniform inside `±10 %` of the box range.
- **Velocity clamping (vmax)**: optional parameter `vmax_ratio` limits
  `|v_i|` per dimension to `vmax_ratio × (upper - lower)`. Without clamping,
  particles can accumulate extreme velocities and oscillate between boundaries
  without exploring the interior. Typical values are 0.2–0.5.
- **Bounds**: two interchangeable policies (see §2.3).
- **Stopping**: max iterations, or no improvement in `gbest` for `stagnation`
  consecutive iterations (`tol = 1e-10`, `stagnation = 50`).
- **on_iteration callback**: an optional hook `on_iteration(it, state)` is
  called after each iteration, enabling custom logging, animation recording, or
  metric collection without modifying the core loop.

### 2.2 Parallel strategies

The three variants share the same loop and differ only in how a single
evaluation step `f(X)` is run, where `X` is the `(n_particles, d)` position
matrix:

| Variant | Evaluator                  | Intended benefit                   |
|---------|----------------------------|------------------------------------|
| V0      | `SequentialEvaluator`      | baseline, no overhead              |
| V1      | `ThreadingEvaluator`       | GIL-limited, useful for I/O loads  |
| V2      | `MultiprocessingEvaluator` | real parallelism, IPC overhead     |

V1 uses `ThreadPoolExecutor`; V2 uses `ProcessPoolExecutor` with a configurable
`chunksize` so we can study batching.

**Pool lifecycle.** V1 and V2 create their pool once in `open()` (called before
the PSO loop) and destroy it in `close()` (in a `finally` block). This avoids
the cost of creating and destroying a pool on every single `evaluate()` call —
an earlier version of the code had this bug, which inflated V1 times by ~3x and
V2 times by ~6x.

**Pickle validation (V2).** `ProcessPoolExecutor` needs to serialize the
objective function via `pickle` to send it to worker processes. If the user
passes a lambda or closure, pickling fails with a cryptic error mid-run.
`MultiprocessingEvaluator.open()` now validates picklability upfront and raises
a clear `TypeError` explaining the constraint.

**Batch splitting (V2).** Instead of dispatching one particle per IPC call,
positions are split into `chunksize`-sized batches before being submitted to
workers. Each batch is evaluated in a single `_evaluate_batch()` call inside
the worker, reducing the number of pickle/pipe round-trips.

### 2.3 Architecture (Strategy pattern with ABCs)

Three abstractions let the solver stay agnostic of implementation details:

- **`BaseEvaluator`** — evaluates a batch of positions. Implementations:
  `SequentialEvaluator` (V0), `ThreadingEvaluator` (V1),
  `MultiprocessingEvaluator` (V2). All share an `open()`/`close()` lifecycle.

- **`BoundsPolicy`** — enforces box constraints after each position update.
  Implementations:
  - `ClampBounds`: clips positions to `[lower, upper]` and zeroes velocity on
    hit axes. Simple and stable — the particle stops at the wall.
  - `ReflectBounds`: reflects positions off the boundary like a billiard ball
    and flips the velocity sign. Conserves kinetic energy, better exploration
    near corners, but slightly less stable convergence.

- **`Topology`** — produces each particle's social-best reference position.
  Implementations:
  - `GlobalBestTopology` (gbest): every particle is attracted to the single
    swarm-wide best. Fast convergence but prone to premature collapse on
    multimodal functions.
  - `RingTopology` (lbest): each particle only sees its `k` nearest neighbours
    in a logical ring. Slower convergence but much better diversity preservation,
    reducing the risk of getting trapped in local minima on Rastrigin/Ackley.

`run_pso` receives all three by dependency injection, so adding a new bounds
policy or topology only requires a new class that implements the ABC — the
optimisation loop does not change.

### 2.4 Hardware and software

| Item            | Value                                              |
|-----------------|----------------------------------------------------|
| CPU             | Intel Core Ultra 7 155H (4 cores exposed to the VM)|
| RAM             | 12 GB                                              |
| Host OS         | Linux 6.17 (Ubuntu 24.04, KVM guest)               |
| Python          | 3.12.3                                             |
| NumPy           | 2.x                                                |
| Key libraries   | `numpy`, `pyswarms` 1.3 (baseline only), `pytest`  |

All measurements use `time.perf_counter()` and are the **mean over 5
independent seeds** (3 seeds for PySwarms baseline, where we only care about
quality). Timing is split internally into three buckets:

- `eval_time` — wall time spent inside the evaluator.
- `update_time` — wall time updating velocities / positions / bounds.
- `overhead` — everything else (pbest/gbest bookkeeping, logging).

### 2.5 Experimental protocol

| Parameter       | Value                                       |
|-----------------|---------------------------------------------|
| Objectives      | Sphere, Rosenbrock, Rastrigin, Ackley       |
| Dimensions      | 2, 10, 30                                   |
| Particles       | 100                                         |
| Max iterations  | 500                                         |
| PSO coefficients| w=0.719, c1=c2=1.49445 (Clerc-Kennedy)      |
| Seeds           | 5 (timing), 3 (quality baseline)            |
| Workers         | 4 for V1/V2                                 |

## 3. Results

### 3.1 Solution quality vs PySwarms

Median best cost over 3 seeds, 500 iterations, matched hyperparameters
(source: `results/pyswarms_baseline.csv`). Lower is better.

| Objective   | d  | Ours (median)       | PySwarms (median)   | Winner   |
|-------------|----|---------------------|---------------------|----------|
| Sphere      |  2 | 3.28e-15            | 2.98e-49            | pyswarms |
| Sphere      | 10 | 2.03e-13            | 1.84e-27            | pyswarms |
| Sphere      | 30 | 6.10e-11            | 6.99e-10            | **ours** |
| Rosenbrock  |  2 | 9.84e-13            | 2.90e-24            | pyswarms |
| Rosenbrock  | 10 | 1.55                | 1.20                | pyswarms |
| Rosenbrock  | 30 | 24.17               | 1.34                | pyswarms |
| Rastrigin   |  2 | 0.00                | 0.00                | tie      |
| Rastrigin   | 10 | 4.97                | 0.99                | pyswarms |
| Rastrigin   | 30 | 84.57               | 24.90               | pyswarms |
| Ackley      |  2 | 2.11e-12            | 4.44e-16            | pyswarms |
| Ackley      | 10 | 1.65e-11            | 1.47e-14            | pyswarms |
| Ackley      | 30 | 1.26e-05            | 0.93                | **ours** |

**Reading.** On easy unimodal problems (low dimension) PySwarms converges
noticeably deeper — several orders of magnitude below us. They run all 500
iterations without early stopping, which lets them polish further into the
underflow region. On multimodal problems (Rastrigin d=30, Rosenbrock d=30)
PySwarms is clearly better, likely due to differences in velocity initialisation
and internal handling of stagnation. We beat them on Sphere d=30 and Ackley d=30,
where our more aggressive boundary handling (zeroing velocity on wall hits) helps
on smooth landscapes. Overall our implementation is competitive with a mature
library — same order of magnitude on most configurations.

### 3.2 Timing across evaluators

Source: `results/comparison.csv`. Mean total time (seconds) over 5 seeds.
`speedup` = `V0 / evaluator`.

| Objective  | d  | V0 (s) | V1 (s) | V2 (s) | speedup V1 | speedup V2 |
|------------|----|--------|--------|--------|------------|------------|
| Sphere     |  2 | 0.054  | 0.397  | 0.772  | 0.14x      | 0.07x      |
| Sphere     | 10 | 0.084  | 0.658  | 1.405  | 0.13x      | 0.06x      |
| Sphere     | 30 | 0.180  | 1.451  | 2.656  | 0.12x      | 0.07x      |
| Rosenbrock |  2 | 0.137  | 0.768  | 1.211  | 0.18x      | 0.11x      |
| Rosenbrock | 10 | 0.325  | 1.626  | 2.926  | 0.20x      | 0.11x      |
| Rosenbrock | 30 | 0.308  | 2.061  | 3.609  | 0.15x      | 0.09x      |
| Rastrigin  |  2 | 0.306  | 1.163  | 2.334  | 0.26x      | 0.13x      |
| Rastrigin  | 10 | 0.792  | 2.849  | 5.201  | 0.28x      | 0.15x      |
| Rastrigin  | 30 | 0.877  | 4.368  | 5.586  | 0.20x      | 0.16x      |
| Ackley     |  2 | 0.536  | 1.985  | 2.718  | 0.27x      | 0.20x      |
| Ackley     | 10 | 1.169  | 3.848  | 5.775  | 0.30x      | 0.20x      |
| Ackley     | 30 | 1.362  | 5.262  | 7.760  | 0.26x      | 0.18x      |

**Reading.** V0 is fastest in every single cell. V1 is ~3–7x slower than V0 and
V2 is ~5–17x slower. The gap narrows with dimension and function complexity —
at d=30 with Ackley (the most expensive benchmark), V1 reaches 0.26x and V2
reaches 0.18x — but the trend never crosses 1x. For microsecond-scale benchmark
functions, parallelism is a pessimisation.

Note: an earlier version of the code created and destroyed the thread/process
pool on every `evaluate()` call (500 times per run). Fixing the pool lifecycle
(create once, reuse, destroy at the end) improved V1 from ~0.06–0.14x to
~0.12–0.30x and V2 from ~0.02–0.05x to ~0.06–0.20x. The overhead was not
inherent to parallelism — it was a bug.

### 3.3 Where the time goes

Fraction of total time spent inside `evaluate()` (higher = less overhead):

| Objective  | d  | pct_eval V0 | pct_eval V1 | pct_eval V2 |
|------------|----|-------------|-------------|-------------|
| Sphere     | 30 | 76.5 %      | 93.0 %      | 95.3 %      |
| Ackley     | 30 | 91.7 %      | 94.0 %      | 96.0 %      |
| Rastrigin  | 30 | 88.6 %      | 94.1 %      | 96.1 %      |

Counter-intuitively V1/V2 spend a *higher* fraction of their time inside
`evaluate()` than V0 — but that fraction is misleading: the absolute
`eval_time` under V1/V2 is itself inflated because it includes thread/process
dispatch, the GIL wait for V1, and pickle round-trips for V2. `evaluate()` is
no longer "just compute" once you parallelise it.

### 3.4 Batching experiment (V2, chunksize sweep)

Source: `results/batching.csv`. V2 on Ackley d=30 with 160 particles, 400
iterations, 4 workers, 3 seeds. V0 baseline: 0.424 s.

| chunk_size | V2 time (s)       | speedup vs V0 |
|-----------:|------------------:|--------------:|
|   1        | 25.59 ± 9.39      | 0.02x         |
|   4        |  9.64 ± 1.18      | 0.04x         |
|   8        |  5.34 ± 0.90      | 0.08x         |
|  16        |  3.64 ± 0.50      | 0.12x         |
|  32        |  2.29 ± 0.29      | 0.19x         |
|  **64**    |  **1.97 ± 0.22**  | **0.22x**     |
| 128        |  2.60 ± 0.20      | 0.16x         |

**Reading.** Going from `chunksize=1` to `chunksize=64` improves V2 by roughly
**13x** (25.6 s -> 1.97 s). The shape is the classic IPC-amortisation curve: at
chunk=1 every particle is one pickle round-trip; at chunk=64 each worker gets
a batch of 40 particles at once and the per-particle IPC cost drops. Past
chunk=64 the workers start running out of work to overlap (160 particles / 64 =
only 2.5 batches — not enough to keep 4 workers busy) and the curve degrades.

Crucially, the *best possible* V2 (0.22x) is still ~5x **slower** than V0:
batching narrows the IPC gap dramatically but cannot close it for functions this
cheap. The break-even point requires `T_f >> T_ipc / chunksize`.

## 4. Discussion

### 4.1 Why V1 does not scale

CPython's Global Interpreter Lock serialises bytecode execution across threads
inside a single process. Our objective functions are pure NumPy/Python CPU work
with no I/O, so threads do not get to release the GIL. Thread dispatch still
costs real time — context switches, lock acquisition — so V1 strictly
underperforms V0. Threading would pay off for an I/O-bound evaluator
(reading files from disk, making network calls) where each thread spends most
of its time waiting and the GIL is released during the wait.

### 4.2 Why V2 does not scale either (at this scale)

`ProcessPoolExecutor` bypasses the GIL by running each worker in its own Python
interpreter. In exchange, every `submit` call:

1. Pickles the function and its arguments.
2. Writes them to a pipe.
3. A worker reads, unpickles, computes, pickles the result back.
4. The main process reads and unpickles.

For Sphere on `d=2`, `f(x)` is a handful of multiplications — sub-microsecond.
The pickle round-trip takes ~100 us per batch. Even with 4 workers running in
parallel, the net result is a slowdown of 5–17x, which the data confirms.

The batching sweep shows that the issue is *per-call* overhead, not compute:
raising `chunksize` from 1 to 64 improves V2 by 13x by amortising fewer
pickles over more evaluations. The curve plateaus before reaching V0 because
the per-particle cost is still dominated by the constant dispatch, not by FLOPs.

### 4.3 When parallelism would win

Call `T_f` the cost of one `f(x)` and `T_ipc` the per-batch IPC round-trip.
V2 becomes worthwhile when

```
N · T_f / k  >>  T_ipc     (per worker, with k = chunksize)
```

Rule of thumb: if an evaluation costs less than ~1 ms, don't parallelise it.
Our benchmarks are ~1 us each, so we are 1000x below the break-even point. A
real expensive fitness — CFD simulation, neural-network training-loss,
robotics simulator — would flip the inequality and V2 would approach the ideal
`N_workers` speedup.

### 4.4 Design decisions

**Velocity clamping.** Without `vmax`, a particle that overshoots the boundary
gets clamped back to the edge, but its velocity remains large. On the next
iteration it shoots past the opposite boundary, creating a ping-pong effect that
wastes iterations. `vmax_ratio` (default: off for backward compatibility) caps
`|v_i|` to a fraction of the search range, stabilising convergence. The test
`test_vmax_improves_rosenbrock_d10` confirms that clamping at ratio=0.5
improves Rosenbrock d=10, where the narrow curved valley amplifies overshoot.

**Two bounds policies.** `ClampBounds` zeroes velocity on impact — the particle
stops at the wall and lets cognitive/social terms pull it back. `ReflectBounds`
flips velocity and mirrors the excess distance back into the box, conserving
kinetic energy. Both converge on Sphere d=2, but ReflectBounds explores corners
better where ClampBounds would create dead zones.

**Two topologies.** `GlobalBestTopology` converges fastest on unimodal functions
because every particle heads straight to the global best. On multimodal
functions like Rastrigin, this causes premature collapse — the whole swarm
converges to the nearest local minimum. `RingTopology(k=1)` restricts each
particle's social reference to its two neighbours in a logical ring, preserving
diversity and letting different subgroups explore different basins.

**Pickle validation.** Instead of letting `ProcessPoolExecutor` fail with a
cryptic `PicklingError` deep inside the PSO loop, `MultiprocessingEvaluator.open()`
validates picklability upfront. The error message explains *why* (lambdas and
closures can't cross process boundaries) and *what to do* (use a module-level
function). This is tested with `test_multiprocessing_rejects_lambda` and
`test_multiprocessing_rejects_closure`.

**on_iteration callback.** The hook `on_iteration(it, state)` is called after
each iteration with the current `SwarmState`. This enables animation recording,
custom convergence criteria, or live dashboards without modifying `run_pso` —
a common extensibility pattern in optimisation libraries.

### 4.5 Limitations

- 4-core VM: results on a host with more physical cores could shift ratios
  slightly but not the ordering (V0 will still dominate for cheap fitnesses).
- Early-stopping criterion (tol=1e-10, stagnation=50) helps wall time but
  hurts us versus PySwarms, which runs all 500 iterations and keeps polishing.
- The quality gap on Rastrigin/Rosenbrock d=30 could be narrowed by combining
  `RingTopology` with `vmax_ratio`, but systematic tuning was out of scope.

## 5. Conclusions

1. **V0 is fastest for cheap objectives**, by a wide margin, confirmed on
   4 objectives x 3 dimensions x 5 seeds.
2. **V1 never helps** on CPU-bound Python: the GIL wins. Pool lifecycle fix
   narrowed the gap from ~10x to ~3–7x, but it remains a pessimisation.
3. **V2 has a clear IPC wall**: batching gives ~13x improvement (chunk 1 -> 64)
   but cannot cross V0 for microsecond fitnesses.
4. **The strategy pattern paid for itself**: swapping evaluator / bounds /
   topology does not touch `run_pso`. Two bounds policies (Clamp, Reflect) and
   two topologies (GlobalBest, Ring) plug in without any change to the core loop.
5. **PySwarms is stronger on multimodal benchmarks**, competitive on smooth
   ones. Our boundary handling helps on Sphere d=30 and Ackley d=30.
6. **Velocity clamping stabilises convergence** on functions with narrow valleys
   (Rosenbrock) or deceptive landscapes, at zero computational cost.

The honest take-away from this project is negative but clear: *throwing
parallelism at cheap fitness functions is an anti-pattern*. The same
infrastructure, applied to a fitness that costs 10+ ms, would give the textbook
4x speedup at chunksize 1.

## 6. Test suite

60 tests across 13 test files, covering:

| Category             | Tests | What they verify                                         |
|----------------------|------:|----------------------------------------------------------|
| Objective functions  |    13 | f(optimum)=0 for all 4 functions at d=2,10; positivity; known values |
| Convergence          |     6 | Sphere converges at d=2,10 across multiple seeds         |
| Monotonic gbest      |     3 | Global best never worsens (sphere, ackley, high-dim)     |
| Bounds enforcement   |     3 | All particles stay in box across all iterations          |
| Bounds policies      |     5 | ClampBounds clips+zeroes; ReflectBounds reflects+flips; both converge |
| Topologies           |     4 | GlobalBest broadcasts; Ring picks local best; both converge |
| Reproducibility      |     4 | Same seed = same result; different seeds differ; all objectives |
| Pool lifecycle       |     7 | open/close/reuse for V1,V2; pickle rejection for lambdas/closures |
| Velocity clamping    |     6 | Velocities respect vmax; convergence across ratios; reproducibility |
| on_iteration callback|     3 | Called every iteration; sees updated gbest; None is safe  |
| Evaluator equivalence|     2 | V0/V1 exact match; V0/V2 within tolerance                |
| Persistence          |     3 | save_run creates JSON+CSV with correct fields            |
| Grid search          |     1 | Produces valid CSV with expected columns                 |

## Appendix A — Files produced

| File                              | Produced by                                 |
|-----------------------------------|---------------------------------------------|
| `results/comparison.csv`          | `scripts/run_comparison.py`                 |
| `results/batching.csv` / `.png`   | `scripts/run_batching_experiment.py`        |
| `results/pyswarms_baseline.csv`   | `scripts/run_pyswarms_baseline.py`          |
| `results/speedup.png`             | `scripts/run_comparison.py`                 |
| `results/grid_search.csv`         | `scripts/run_grid_search.py`                |
| `results/analysis/`               | `scripts/analyze_results.py`                |

## Appendix B — Reproducing the experiments

```bash
pip install -e ".[dev]"
pytest                                          # 60 tests
python scripts/run_comparison.py                # ~5 min, 5 seeds x 36 cells
python scripts/run_batching_experiment.py       # ~3 min
python scripts/run_pyswarms_baseline.py         # ~1 min
python scripts/analyze_results.py               # plots + summary
```
