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

Canonical inertia-weight PSO with global-best topology. Each iteration:

```
v_i ← w · v_i + c1 · r1 · (pbest_i − x_i) + c2 · r2 · (gbest − x_i)
x_i ← x_i + v_i
```

- **Initialisation**: positions uniform inside `[lower, upper]^d`,
  velocities uniform inside `±10 %` of the box range.
- **Bounds**: clamp positions and zero the velocity component on any axis that
  hit the wall (prevents particles bouncing back and forth on a boundary).
- **Stopping**: max iterations, or no improvement in `gbest` for `stagnation`
  consecutive iterations (`tol = 1e-10`, `stagnation = 50`).

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

### 2.3 Architecture (Strategy pattern with ABCs)

Three abstractions let the solver stay agnostic of implementation details:

- `BaseEvaluator` — evaluates a batch of positions (`sequential` / `threading`
  / `multiprocessing`).
- `BoundsPolicy` — enforces box constraints (`ClampBounds`).
- `Topology` — produces each particle's social best
  (`GlobalBestTopology`).

`run_pso` receives all three by dependency injection, so adding a new bounds
policy (e.g. reflection) or topology (e.g. ring) only requires a new class that
implements the ABC — the optimisation loop does not change.

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
| PSO coefficients| w=0.719, c1=c2=1.49445 (Clerc–Kennedy)      |
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
noticeably deeper — a few orders of magnitude below us. They stop at the
underflow region while our early-stopping cuts us off earlier. On hard,
multimodal problems (Rastrigin, Rosenbrock at d=30) PySwarms is clearly better,
suggesting their exploration behaviour — slightly different initialisation of
velocities and no velocity clamping after boundary hits — helps them escape
local minima. We beat them on Sphere d=30 and Ackley d=30, which are smooth
enough to benefit from our more aggressive wall behaviour. Overall our
implementation is competitive but not state-of-the-art.

### 3.2 Timing across evaluators

Source: `results/comparison.csv`. Mean total time (seconds) over 5 seeds.
`speedup` = `V0 / evaluator`.

| Objective  | d  | V0 (s) | V1 (s) | V2 (s) | speedup V1 | speedup V2 |
|------------|----|--------|--------|--------|------------|------------|
| Sphere     |  2 | 0.07   | 1.17   | 4.83   | 0.06x      | 0.02x      |
| Sphere     | 10 | 0.08   | 1.15   | 4.84   | 0.07x      | 0.02x      |
| Sphere     | 30 | 0.32   | 3.30   | 12.93  | 0.10x      | 0.03x      |
| Rosenbrock |  2 | 0.12   | 0.93   | 3.92   | 0.13x      | 0.03x      |
| Rosenbrock | 10 | 0.27   | 2.38   | 9.68   | 0.11x      | 0.03x      |
| Rosenbrock | 30 | 0.29   | 2.63   | 11.51  | 0.11x      | 0.03x      |
| Rastrigin  |  2 | 0.07   | 0.65   | 2.57   | 0.10x      | 0.03x      |
| Rastrigin  | 10 | 0.19   | 1.75   | 6.93   | 0.11x      | 0.03x      |
| Rastrigin  | 30 | 0.25   | 2.20   | 8.98   | 0.11x      | 0.03x      |
| Ackley     |  2 | 0.15   | 1.11   | 4.16   | 0.14x      | 0.04x      |
| Ackley     | 10 | 0.25   | 1.87   | 7.04   | 0.13x      | 0.04x      |
| Ackley     | 30 | 0.37   | 2.74   | 9.93   | 0.13x      | 0.04x      |

**Reading.** V0 is fastest in every single cell. V1 is ~10x slower than V0 and
V2 is ~30x slower. Speedups mildly improve with dimension — at d=30 the
evaluation itself is slightly more expensive, so the parallel overhead eats a
smaller fraction of total time — but the trend never crosses 1x. For
microsecond-scale benchmark functions, parallelism is a pessimisation.

### 3.3 Where the time goes

Fraction of total time spent inside `evaluate()` (higher = less overhead):

| Objective  | d  | pct_eval V0 | pct_eval V1 | pct_eval V2 |
|------------|----|-------------|-------------|-------------|
| Sphere     | 30 | 81.5 %      | 96.4 %      | 98.0 %      |
| Ackley     | 30 | 89.0 %      | 96.6 %      | 98.2 %      |
| Rastrigin  | 30 | 86.3 %      | 96.4 %      | 98.1 %      |

Counter-intuitively V1/V2 spend a *higher* fraction of their time inside
`evaluate()` than V0 — but that fraction is misleading: the absolute
`eval_time` under V1/V2 is itself inflated because it includes thread/process
dispatch, the GIL wait for V1, and pickle round-trips for V2. `evaluate()` is
no longer "just compute" once you parallelise it.

### 3.4 Batching experiment (V2, chunksize sweep)

Source: `results/batching.csv`. V2 on Ackley d=30 with 160 particles, 400
iterations, 4 workers, 3 seeds. V0 baseline: 0.358 s.

| chunk_size | V2 time (s) | speedup vs V0 |
|-----------:|------------:|--------------:|
|   1        | 26.00 ± 2.53| 0.01x         |
|   4        | 10.84 ± 0.10| 0.03x         |
|   8        |  8.67 ± 0.14| 0.04x         |
|  16        |  7.56 ± 0.05| 0.05x         |
|  32        |  6.87 ± 0.12| 0.05x         |
|  **64**    |  **6.71 ± 0.02**| **0.05x** |
| 128        |  7.06 ± 0.19| 0.05x         |

**Reading.** Going from `chunksize=1` to `chunksize=64` improves V2 by roughly
**4x** (26 s → 6.7 s). The shape is the classic IPC-amortisation curve: at
chunk=1 every particle is one pickle round-trip; at chunk=64 each worker gets
40 particles at once and the per-particle IPC cost drops. Past chunk=64 the
workers start running out of work to overlap and the curve flattens / slightly
degrades. Crucially, the *best possible* V2 (0.05x) is still 20x **slower**
than V0: batching narrows the IPC gap but cannot cross it for functions this
cheap.

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
interpreter. In exchange, every `map` call:

1. Pickles the function arguments.
2. Writes them to a pipe.
3. A worker reads, unpickles, computes, pickles the result back.
4. The main process reads and unpickles.

For Sphere on `d=2`, `f(x)` is a handful of multiplications — sub-microsecond.
The pickle round-trip takes ~100 µs per batch. Even with 4 workers running in
parallel, the net result is a slowdown of 30–60x, exactly what the data shows.

The batching sweep confirms that the issue is *per-call* overhead, not compute:
raising `chunksize` from 1 to 64 improves V2 4x by amortising a smaller number
of pickles over a larger number of evaluations. The curve plateaus before
reaching V0 because the per-particle cost is still dominated by the constant
dispatch, not by the FLOPs.

### 4.3 When parallelism would win

Call `T_f` the cost of one `f(x)` and `T_ipc` the per-batch IPC round-trip.
V2 becomes worthwhile when

```
N · T_f / k  >>  T_ipc     (per worker, with k = chunksize)
```

Rule of thumb: if an evaluation costs less than ~1 ms, don't parallelise it.
Our benchmarks are ~1 µs each, so we are 1000x below the break-even point. A
real expensive fitness — CFD simulation, neural-network training-loss,
robotics simulator — would flip the inequality and V2 would approach the ideal
`N_workers` speedup.

### 4.4 Limitations

- 4-core VM: results on a host with more physical cores could shift ratios
  slightly but not the ordering (V0 will still dominate for cheap fitnesses).
- Single topology (global-best) and single bounds policy (clamp): other
  topologies like ring explore more and might narrow the quality gap with
  PySwarms on multimodal problems.
- The early-stopping criterion (tol=1e-10, stagnation=50) helps wall time but
  hurts us versus PySwarms, which runs all 500 iterations and keeps polishing.

## 5. Conclusions

1. **V0 is fastest for cheap objectives**, by a wide margin, confirmed on
   4 objectives × 3 dimensions × 5 seeds.
2. **V1 never helps** on CPU-bound Python: the GIL wins.
3. **V2 has a clear IPC wall**: batching gives ~4x improvement but cannot
   cross V0 for microsecond fitnesses.
4. **The strategy pattern paid for itself**: swapping evaluator / bounds /
   topology does not touch `run_pso`, and the batching experiment and PySwarms
   baseline reused the same optimisation loop.
5. **PySwarms is stronger on multimodal benchmarks**, competitive on smooth
   ones. Our boundary handling helps on Sphere d=30 and Ackley d=30.

The honest take-away from this project is negative but clear: *throwing
parallelism at cheap fitness functions is an anti-pattern*. The same
infrastructure, applied to a fitness that costs 10+ ms, would give the textbook
4x speedup at chunksize 1.

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
source zonaproyecto/bin/activate
pytest                                          # 11 tests
python scripts/run_comparison.py                # ~15 min, 5 seeds × 36 cells
python scripts/run_batching_experiment.py       # ~2 min
python scripts/run_pyswarms_baseline.py         # ~1 min
python scripts/analyze_results.py               # plots + summary
```
