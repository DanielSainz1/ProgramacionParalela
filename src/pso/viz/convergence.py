import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_convergence(
    histories: list,
    labels: list,
    title: str = "PSO Convergence",
    out_path: str = None,
) -> None:
    """Plot convergence curves (best fitness vs iteration) in log scale."""
    plt.figure(figsize=(10, 6))

    for history, label in zip(histories, labels):
        plt.plot(history, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Best value (log scale)")
    plt.title(title)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Plot saved to: %s", out_path)
    else:
        plt.show()
