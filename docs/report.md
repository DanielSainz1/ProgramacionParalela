# PSO Project — Report

## 1. Introduction

This project implements PSO (Particle Swarm Optimization) with three evaluation strategies: sequential (V0), threading (V1), and multiprocessing (V2). We compare them on 4 benchmark functions in dimensions 2, 10 and 30.

## 2. Methodology

### PSO algorithm

We use the canonical PSO with global-best topology. Each iteration updates velocities and positions using the standard equations with inertia weight w, cognitive coefficient c1, and social coefficient c2. Boundary handling is done with clamping (np.clip). The algorithm stops either at max_iter, or early if gbest doesn't improve for `stagnation` consecutive iterations.

### Parallel strategies

All three strategies use the same PSO loop — only the evaluator changes (strategy pattern):

- **V0**: Simple for-loop. No overhead.
- **V1**: ThreadPoolExecutor with configurable workers. Doesn't help for CPU-bound code because of the GIL.
- **V2**: ProcessPoolExecutor with configurable workers and chunksize for batching. Real parallelism but IPC/pickle overhead.

### Experimental setup

- Functions: Sphere, Rosenbrock, Rastrigin, Ackley
- Dimensions: 2, 10, 30
- Parameters: w=0.719, c1=1.49445, c2=1.49445, 100 particles, 500 max iterations
- 5 seeds per configuration
- Timing with time.perf_counter()

## 3. Results

### Solution quality

All three strategies give the same result for the same seed. This is verified by tests (test_evaluator_equivalence.py). executor.map() preserves order so reproducibility is maintained.

### Timing

For our benchmark functions (microsecond evaluation), V0 is always fastest. V1 and V2 are much slower because of overhead:

- V1: GIL prevents real parallelism + thread creation/context-switch overhead
- V2: pickle serialization + IPC transfer costs more than the evaluation itself

The timing breakdown in PSOResult shows that for V0, eval_time is ~90% of total. For V1/V2, the overhead dominates.

### Grid search

We ran a 3x3x3 grid over w, c1, c2 with 5 seeds. The Clerc-Kennedy values (w=0.719, c1=c2=1.49445) perform well across all functions as expected from the literature.

## 4. Discussion

### GIL (V1)

The Global Interpreter Lock only lets one thread execute Python bytecode at a time. For CPU-bound code like our objective functions, threads just take turns instead of running in parallel. The context-switching overhead makes it slower than sequential. Threading would help for I/O-bound tasks where the GIL is released during waits.

### IPC overhead (V2)

Multiprocessing gives real parallelism (each process has its own GIL) but needs to serialize data with pickle and send it through pipes. For a function that takes 1 microsecond, the serialization round-trip takes ~100 microseconds — 100x more than the computation. Batching with chunksize reduces the number of IPC calls but doesn't fully compensate for cheap functions.

### When would parallelism help?

If the objective function was expensive (>1ms per evaluation, like a simulation or ML model), the computation time would dominate over IPC cost and V2 would give real speedup. For our benchmarks it's not worth it, which is expected.

## 5. Conclusion

- V0 is fastest for cheap functions (no overhead)
- V1 doesn't help because of the GIL
- V2 would help for expensive objectives but not for our microsecond benchmarks
- The strategy pattern makes it easy to switch between evaluators
- All strategies give identical results for the same seed
