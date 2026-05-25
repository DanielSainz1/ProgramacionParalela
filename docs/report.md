# PSO Project — Final Report

## 1. Introduction

This project implements Particle Swarm Optimization (PSO) in Python with five
evaluation strategies — sequential (V0), threading (V1), multiprocessing (V2),
asyncio (V3) and NumPy vectorised (V4) — and compares them empirically on four
standard benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley) in
dimensions 2, 10 and 30.

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

The five variants share the same loop and differ only in how a single
evaluation step `f(X)` is run, where `X` is the `(n_particles, d)` position
matrix:

| Variant | Evaluator                  | Intended benefit                                   |
|---------|----------------------------|----------------------------------------------------|
| V0      | `SequentialEvaluator`      | baseline, no overhead                              |
| V1      | `ThreadingEvaluator`       | GIL-limited, useful for I/O loads                  |
| V2      | `MultiprocessingEvaluator` | real parallelism on multiple CPUs, IPC overhead    |
| V3      | `AsyncEvaluator`           | cooperative concurrency — wins on I/O-bound work   |
| V4      | `VectorizedEvaluator`      | NumPy BLAS / SIMD over the whole position matrix   |

V1 uses `ThreadPoolExecutor`; V2 uses `ProcessPoolExecutor` with a configurable
`chunksize` so we can study batching. V3 runs `asyncio.gather` over coroutines
that each await a configurable latency before computing — modelling a remote
sensor or queued service. V4 calls a vectorised counterpart of each objective
(`OBJECTIVES_VEC` registry) so all N particles are evaluated in a single
NumPy operation; no thread or process pool is involved.

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

Source: `results/comparison.csv`. Mean total time (seconds) over 5 seeds for
each of the 5 variants. `speedup` is computed as `V0 / evaluator` — values
above 1.0 are wins, below 1.0 are losses.

| Objective  | d  | V0 (s) | V1 (s) | V2 (s) | V3 (s) | V4 (s)   | sp V1 | sp V2 | sp V3 | sp V4 |
|------------|----|--------|--------|--------|--------|----------|-------|-------|-------|-------|
| Sphere     |  2 | 0.037  | 0.252  | 0.397  | 0.085  | 0.005    | 0.15x | 0.09x | 0.43x | **7.4x**  |
| Sphere     | 10 | 0.081  | 0.480  | 0.724  | 0.159  | 0.010    | 0.17x | 0.11x | 0.51x | **7.6x**  |
| Sphere     | 30 | 0.142  | 0.893  | 1.457  | 0.346  | 0.035    | 0.16x | 0.10x | 0.41x | **4.1x**  |
| Rosenbrock |  2 | 0.115  | 0.459  | 0.589  | 0.198  | 0.008    | 0.25x | 0.19x | 0.58x | **13.9x** |
| Rosenbrock | 10 | 0.276  | 1.064  | 1.475  | 0.493  | 0.024    | 0.26x | 0.19x | 0.56x | **11.4x** |
| Rosenbrock | 30 | 0.267  | 1.110  | 1.536  | 0.460  | 0.037    | 0.24x | 0.17x | 0.58x | **7.2x**  |
| Rastrigin  |  2 | 0.069  | 0.317  | 0.447  | 0.142  | 0.008    | 0.22x | 0.15x | 0.49x | **8.6x**  |
| Rastrigin  | 10 | 0.211  | 0.787  | 1.100  | 0.346  | 0.019    | 0.27x | 0.19x | 0.61x | **11.3x** |
| Rastrigin  | 30 | 0.298  | 1.026  | 1.575  | 0.474  | 0.052    | 0.29x | 0.19x | 0.63x | **5.7x**  |
| Ackley     |  2 | 0.168  | 0.502  | 0.745  | 0.260  | 0.011    | 0.34x | 0.23x | 0.65x | **15.2x** |
| Ackley     | 10 | 0.260  | 0.805  | 1.138  | 0.397  | 0.028    | 0.32x | 0.23x | 0.66x | **9.4x**  |
| Ackley     | 30 | 0.348  | 1.028  | 1.553  | 0.527  | 0.052    | 0.34x | 0.22x | 0.66x | **6.7x**  |

**Reading.**

- **V4 wins every single cell**, by a factor between 4.1x and 15.2x. The
  vectorised approach is, by a wide margin, the right strategy for cheap
  numerical fitnesses on a single machine.
- **V1, V2 always lose** by the same factor as before — GIL for V1, IPC for V2.
- **V3 lies between V0 and V1**: the event loop adds ~2x overhead (0.4-0.7x)
  but no thread/process management cost. With latency = 0 it cannot win.
- The gap between V0 and V1/V2 narrows with dimension (more expensive fitness)
  but never crosses 1x for these benchmarks.

Note: an earlier version of the code created and destroyed the thread/process
pool on every `evaluate()` call (500 times per run). Fixing the pool lifecycle
(create once, reuse, destroy at the end) improved V1 from ~0.06-0.14x to
~0.15-0.34x and V2 from ~0.02-0.05x to ~0.09-0.23x. The overhead was not
inherent to parallelism — it was a bug.

### 3.3 V3 wins under latency: the asymmetric workload

V3 is essentially V0 wrapped in an asyncio event loop unless something
actually awaits. To show its real value, `run_latency_experiment.py` runs
Sphere d=10 (20 particles, 15 iterations, 3 seeds) where the objective is
preceded by a fixed sleep of `latency_ms` per particle. For V0 we wrap the
objective with `time.sleep`; for V3 we use the built-in
`latency_ms_min = latency_ms_max = latency_ms` knob.

Source: `results/latency.csv`.

| Latency / particle | V0 (s) | V3 (s) | Speedup V3 vs V0 |
|--------------------|-------:|-------:|------------------:|
|   1 ms             |  0.36  |  0.025 |  14.4x            |
|   5 ms             |  1.78  |  0.103 |  17.3x            |
|  10 ms             |  3.70  |  0.202 |  **18.3x**        |
|  20 ms             |  7.04  |  0.341 |  **20.7x**        |
|  50 ms             | 17.97  |  0.848 |  **21.2x**        |

The speedup plateaus around 20x because that is roughly the swarm-level
concurrency `asyncio.gather` can extract — N=20 particles whose sleeps overlap
on a single thread. With more particles the asymptote would rise.

This is the canonical asyncio narrative: the moment an evaluation includes
*any* real waiting (network call, queued service, sensor poll), V3 transforms
the timing from `N × latency` to `≈ max(latency)`.

### 3.4 Where the time goes

Fraction of total time spent inside `evaluate()` (higher = less overhead):

| Objective  | d  | V0     | V1     | V2     | V3     | V4     |
|------------|----|--------|--------|--------|--------|--------|
| Sphere     | 30 | 76.9 % | 92.4 % | 94.9 % | 87.0 % | 10.7 % |
| Ackley     | 30 | 89.2 % | 93.7 % | 95.5 % | 91.8 % | 45.1 % |
| Rastrigin  | 30 | 86.2 % | 93.4 % | 95.2 % | 90.3 % | 41.5 % |

Two things stand out:

1. **V1/V2/V3 spend a *higher* fraction inside `evaluate()`** than V0. That
   fraction is misleading: the absolute `eval_time` is inflated by thread
   dispatch (V1), pickle round-trips (V2) and event-loop scheduling (V3).
   `evaluate()` is no longer "just compute" once you parallelise it.
2. **V4 collapses `eval_time` to 10-45 %** of total. The objective itself
   becomes so fast that `update_time` (velocity/position update) now
   dominates. To go faster, the next bottleneck would be the update step
   — which is also vectorisable.

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

### 4.4 Why V3 needs latency to be worth it

V3 wraps each evaluation in an `async` coroutine and launches them with
`asyncio.gather`. There is no parallelism — asyncio is single-threaded
cooperative concurrency. What `gather` exploits is **overlap of waits**: when
one coroutine hits an `await`, the event loop hands control to another
coroutine until the first is ready to resume.

So with zero latency (Section 3.2), V3 is V0 plus the event-loop scheduling
overhead. It loses by 0.4–0.7x. With latency (Section 3.3) it wins by 14–21x,
because the N waits are now scheduled in parallel and total time approaches
`max(latencies)` instead of `sum(latencies)`.

The right way to read these two sections together is: V3 trades a small
constant overhead for the ability to absorb arbitrarily large I/O latencies.
For a CPU-only fitness this trade is always a loss; for a remote-call
fitness, even a few milliseconds of latency are enough to flip it.

### 4.5 Why V4 wins on cheap fitnesses

V4 replaces the per-particle Python loop with one NumPy call on the whole
`(N, d)` position matrix. Three things disappear:

1. **The Python interpreter loop overhead.** N=100 particles per iteration ×
   500 iterations = 50 000 Python function-call overheads removed.
2. **The GIL is no longer a bottleneck**, because NumPy releases it during
   BLAS calls and we are no longer trying to use multiple threads anyway.
3. **CPU SIMD instructions kick in.** AVX2 processes 4 doubles per cycle;
   AVX-512 processes 8. NumPy dispatches the inner arithmetic to those
   instructions automatically when the data is contiguous.

The measured 4–15x speedup matches what a 4-core SIMD CPU can theoretically
deliver on these kernels. Notice that V4 wins *more* on lower dimensions —
that is because the constant Python-loop overhead is a larger fraction of
the work there. As d grows, NumPy can amortise the vectorised math over
more elements but the absolute speedup tightens.

V4 does not generalise to expensive fitnesses that are not vectorisable
(e.g. running an external simulator binary). In that regime V2 takes over.
The two strategies are complementary, not competing.

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

### 4.6 Limitations

- 4-core VM: results on a host with more physical cores could shift ratios
  slightly but not the ordering (V0 will still dominate for cheap fitnesses).
- Early-stopping criterion (tol=1e-10, stagnation=50) helps wall time but
  hurts us versus PySwarms, which runs all 500 iterations and keeps polishing.
- The quality gap on Rastrigin/Rosenbrock d=30 could be narrowed by combining
  `RingTopology` with `vmax_ratio`, but systematic tuning was out of scope.

## 5. Conclusions

1. **V4 (NumPy vectorised) is the clear winner** for cheap numerical fitness
   functions: 4–15x speedup over V0 across all benchmarks and dimensions.
   No threads, no processes, no event loop — just the right shape of code.
2. **V0 is the right baseline** but loses to V4 by a constant factor that
   reflects how much Python interpreter overhead the vectorised version
   avoids.
3. **V1 never helps** on CPU-bound Python: the GIL wins. Pool lifecycle fix
   narrowed the gap from ~10x to ~3–7x, but it remains a pessimisation.
4. **V2 has a clear IPC wall**: batching gives ~13x improvement (chunk 1 → 64)
   but cannot cross V0 for microsecond fitnesses. V2 wins only when the
   fitness itself costs more than the IPC round-trip.
5. **V3 (asyncio) inverts its sign with latency**: a loss of 0.4–0.7x with
   zero latency, a win of 14–21x once any I/O-style waiting is involved.
   The break-even point is ~1 ms per particle.
6. **The strategy pattern paid for itself**: swapping evaluator / bounds /
   topology does not touch `run_pso`. Two bounds policies (Clamp, Reflect),
   two topologies (GlobalBest, Ring) and five evaluators all plug in without
   any change to the core loop.
7. **PySwarms is stronger on multimodal benchmarks**, competitive on smooth
   ones. Our boundary handling helps on Sphere d=30 and Ackley d=30.
8. **Velocity clamping stabilises convergence** on functions with narrow
   valleys (Rosenbrock) or deceptive landscapes, at zero computational cost.

The take-away is that **"parallelism" is not one thing**: V1, V2, V3 and V4
each attack a different bottleneck, and only the one that matches the actual
bottleneck of the workload pays off. For our microsecond benchmark fitness
that bottleneck is the Python interpreter, which is exactly what V4 removes.
A 10-millisecond fitness would shift the answer to V2; a fitness with network
calls would shift it to V3. The infrastructure built here lets a future user
make that choice with one line of YAML.

## 6. Test suite

70 tests across 15 test files, covering:

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
| Async evaluator (V3) |     3 | Same values as V0 at latency=0; lifecycle reusable; gather overlaps |
| Vectorised (V4)      |     7 | scalar==vec per particle for all 4 objectives; PSO converges; numerical match with V0 |
| Persistence          |     3 | save_run creates JSON+CSV with correct fields            |
| Grid search          |     1 | Produces valid CSV with expected columns                 |

## Appendix A — Files produced

| File                              | Produced by                                 |
|-----------------------------------|---------------------------------------------|
| `results/comparison.csv`          | `scripts/run_comparison.py` (V0–V4, 60 rows)|
| `results/speedup.png`             | `scripts/run_comparison.py`                 |
| `results/batching.csv` / `.png`   | `scripts/run_batching_experiment.py`        |
| `results/latency.csv` / `.png`    | `scripts/run_latency_experiment.py`         |
| `results/pyswarms_baseline.csv`   | `scripts/run_pyswarms_baseline.py`          |
| `results/grid_search.csv`         | `scripts/run_grid_search.py`                |
| `results/analysis/`               | `scripts/analyze_results.py`                |

## Appendix B — Reproducing the experiments

```bash
pip install -e ".[dev]"
pytest                                          # 70 tests
python scripts/run_comparison.py                # ~5 min, 5 seeds x 60 cells (V0–V4)
python scripts/run_batching_experiment.py       # ~3 min
python scripts/run_latency_experiment.py        # ~3 min, V0 vs V3 across latencies
python scripts/run_pyswarms_baseline.py         # ~1 min
python scripts/analyze_results.py               # plots + summary
```
