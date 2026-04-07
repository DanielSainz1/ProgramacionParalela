import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable

logger = logging.getLogger(__name__)


def animate_swarm_3d(
    position_history: list,
    gbest_history: list,
    objective: Callable,
    lower: float,
    upper: float,
    out_path: str = "results/swarm_3d.gif",
    fps: int = 10,
) -> None:
    """Generate a GIF of the swarm in 3D. Only works for dim=3."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")

    particles_plot = ax.scatter([], [], [], c="white", edgecolors="blue", s=20, alpha=0.7)
    gbest_plot = ax.scatter([], [], [], c="red", marker="*", s=150)
    title = ax.set_title("")

    def update(frame):
        positions = position_history[frame]
        gbest = gbest_history[frame]

        # scatter doesn't support set_data, use _offsets3d
        particles_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        gbest_plot._offsets3d = ([gbest[0]], [gbest[1]], [gbest[2]])
        title.set_text(f"Iteration {frame + 1} / {len(position_history)}")
        return particles_plot, gbest_plot, title

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(position_history),
        interval=1000 // fps,
        blit=False,  # blit=False required for 3D
    )

    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()
    logger.info("3D animation saved to: %s", out_path)
