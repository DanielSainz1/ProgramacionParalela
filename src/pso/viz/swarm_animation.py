import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable

logger = logging.getLogger(__name__)

def animate_swarm_2d(
    position_history: list,
    gbest_history: list,
    objective: Callable,
    lower: float,
    upper: float,
    out_path: str = "results/swarm.gif",
    fps: int = 10,
) -> None:
    
    grid_points = np.linspace(lower, upper, 200)
    X, Y = np.meshgrid(grid_points, grid_points)
    Z = np.array([[objective(np.array([x, y])) for x in grid_points] for y in grid_points])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    
    particles_plot, = ax.plot([], [], "o", color="white", ms=4, alpha=0.7)
    gbest_plot,     = ax.plot([], [], "*", color="red",   ms=12)
    title = ax.set_title("")

    def update(frame):
        positions = position_history[frame]           
        particles_plot.set_data(positions[:, 0], positions[:, 1])
        gbest_plot.set_data([gbest_history[frame][0]], [gbest_history[frame][1]])
        title.set_text(f"Iteración {frame + 1} / {len(position_history)}")
        return particles_plot, gbest_plot, title

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(position_history),
        interval=1000 // fps,
        blit=True,
    )

    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()
    logger.info("Animación guardada en: %s", out_path)
