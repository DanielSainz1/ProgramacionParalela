import argparse
import csv
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_runs(results_dir, objective=None, dim=None, evaluator=None):
    """
    Scan results_dir for saved runs.
    Returns a list of dicts with keys: config, history, run_dir.
    Filters by objective, dim and evaluator if given.
    """
    runs = []
    results_path = Path(results_dir)

    for run_dir in sorted(results_path.iterdir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        pso = config["pso"]

        # Apply filters
        if objective and pso["objective"] != objective:
            continue
        if dim and pso["dim"] != dim:
            continue
        if evaluator and pso.get("evaluator", "sequential") != evaluator:
            continue

        # Load convergence history
        history = []
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(float(row["gbest_value"]))

        runs.append({
            "config": pso,
            "history": history,
            "run_dir": str(run_dir),
        })

    return runs


def plot_convergence_comparison(runs, out_path=None):
    """Plot convergence curves for all loaded runs on the same axes."""
    if not runs:
        logger.warning("No runs found.")
        return

    plt.figure(figsize=(10, 6))

    for run in runs:
        cfg = run["config"]
        label = f"{cfg['objective']} d={cfg['dim']} s={cfg['seed']} ({cfg.get('evaluator', 'seq')})"
        plt.plot(run["history"], label=label, alpha=0.8)

    plt.xlabel("Iteration")
    plt.ylabel("Best value (log scale)")
    plt.title("Convergence comparison")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Convergence plot saved to: %s", out_path)
    else:
        plt.show()


def plot_avg_convergence(runs, out_path=None):
    """Plot average convergence curves grouped by evaluator, with std shading."""
    if not runs:
        logger.warning("No runs found.")
        return

    import numpy as np

    # Group histories by evaluator
    groups = {}
    for run in runs:
        evaluator = run["config"].get("evaluator", "sequential")
        groups.setdefault(evaluator, []).append(run["history"])

    plt.figure(figsize=(10, 6))

    for evaluator, histories in groups.items():
        # Pad shorter histories to the max length with their last value
        max_len = max(len(h) for h in histories)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        iters = np.arange(max_len)

        plt.plot(iters, mean, label=f"{evaluator} (n={len(histories)})")
        plt.fill_between(iters, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Best value (log scale)")
    plt.title("Average convergence by evaluator")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Average convergence plot saved to: %s", out_path)
    else:
        plt.show()


def plot_boxplot(runs, out_path=None):
    """Boxplot of final best fitness grouped by evaluator."""
    if not runs:
        logger.warning("No runs found.")
        return

    # Group final values by evaluator
    groups = {}
    for run in runs:
        evaluator = run["config"].get("evaluator", "sequential")
        final_value = run["history"][-1]
        groups.setdefault(evaluator, []).append(final_value)

    labels = list(groups.keys())
    data = [groups[label] for label in labels]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("Final fitness (log scale)")
    plt.yscale("log")
    plt.title("Final fitness distribution by evaluator")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Boxplot saved to: %s", out_path)
    else:
        plt.show()


def print_summary_table(runs):
    """Print a text summary table of all runs."""
    if not runs:
        logger.warning("No runs found.")
        return

    header = f"\n{'Objective':12s} {'Dim':>4s} {'Evaluator':>14s} {'Seed':>6s} {'Best Value':>14s}"
    logger.info(header)
    logger.info("-" * 56)
    for run in runs:
        cfg = run["config"]
        logger.info("%12s %4d %14s %6d %14.4e",
                     cfg['objective'], cfg['dim'], cfg.get('evaluator', 'seq'),
                     cfg['seed'], run['history'][-1])


def main():
    parser = argparse.ArgumentParser(description="Analyze saved PSO results")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--evaluator", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/analysis/")
    args = parser.parse_args()

    runs = load_runs(args.results_dir, args.objective, args.dim, args.evaluator)
    logger.info("Found %d runs", len(runs))

    if not runs:
        return

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    print_summary_table(runs)

    # Convergence curves (individual)
    plot_convergence_comparison(runs, out_path=str(out_dir / "convergence.png"))

    # Average convergence curves (grouped by evaluator)
    plot_avg_convergence(runs, out_path=str(out_dir / "avg_convergence.png"))

    # Boxplot
    plot_boxplot(runs, out_path=str(out_dir / "boxplot.png"))


if __name__ == "__main__":
    main()
