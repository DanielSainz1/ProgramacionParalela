# Design Document

## Architecture

The project is split into modules that each do one thing:

```
src/pso/
  core/           → PSO algorithm (doesn't know about parallelism)
  eval/           → The 3 evaluation strategies (V0, V1, V2)
  objectives/     → Benchmark functions (sphere, rastrigin, etc.)
  experiments/    → Config loading and orchestration
  io/             → Saving results to disk
  viz/            → Plots and animations

scripts/          → CLI entry points
tests/            → Automated tests
```

### How modules connect

```
scripts/*
  └── experiments/runner.py
        ├── experiments/config.py   (PSOConfig dataclass)
        ├── core/pso.py             (run_pso, PSOResult)
        │     ├── core/state.py     (SwarmState)
        │     ├── core/bounds.py    (clamp_positions)
        │     └── eval/base.py      (BaseEvaluator — injected)
        ├── eval/sequential.py      (V0)
        ├── eval/threading_eval.py  (V1)
        ├── eval/multiprocessing_eval.py (V2)
        ├── objectives/*            (OBJECTIVES registry)
        └── io/persistence.py       (save_run)
```

Dependencies go downward only — no circular imports.

Adding a new evaluator (V3) would just mean creating a new file in eval/ and adding one line to the EVALUATORS dict in runner.py. Nothing else changes.

## Design decisions

**Strategy pattern for evaluators**: run_pso() receives a BaseEvaluator object and just calls evaluate(). It doesn't know or care if it's sequential, threaded or multiprocessing. This keeps the PSO logic clean and makes it easy to swap strategies.

**Clamping for bounds**: We use np.clip to keep particles inside the search space. Simpler than reflection or penalty and works fine for our benchmarks.

**JSON + CSV for persistence**: JSON for the config (it's hierarchical — params, timing, metadata). CSV for per-iteration metrics (easy to load and plot). We considered SQLite but it's overkill for this project.

**YAML + CLI config**: Default parameters in configs/default.yaml, CLI flags override individual values. This way you can experiment quickly without editing files.

**Logging, not print**: All modules use logging.getLogger(__name__). Scripts configure the format. Library code never calls basicConfig() — that's a Python best practice.

## Parallelism trade-offs

| | V0 Sequential | V1 Threading | V2 Multiprocessing |
|---|---|---|---|
| Overhead | None | Thread creation + GIL contention | Process creation + IPC + pickle |
| Real parallelism? | No | No (GIL) | Yes |
| Best for | Cheap functions | I/O-bound work | Expensive CPU-bound functions |
| Workers | N/A | Configurable | Configurable |
| Batching | N/A | N/A | chunksize parameter |

For our benchmarks (microsecond evaluations), V0 wins because there's no overhead. V2 would win if each evaluation took milliseconds or more.

## Known limitations

- Only global-best topology (no neighborhood topologies)
- Only clamping for boundaries (no reflection/penalty)
- Fixed w, c1, c2 (no adaptive parameters)
- Animation only works for d=2 and d=3
- For cheap functions, parallel strategies are slower than sequential (expected)
