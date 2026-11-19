"""Show the source kernel samples over its continuous signed grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import GRID_BITS, Grid


def save_kernel(grid: Grid, path: Path) -> Path:
    """Show all original 64 grid samples on the unchanged finite domain."""
    dense = np.linspace(-grid.radius, grid.radius, 500)
    magnitude = np.exp(-(dense**2) / (4 * grid.gamma**2)) / (1 + dense**2)
    samples = np.exp(-(grid.points**2) / (4 * grid.gamma**2)) / (1 + grid.points**2)
    figure, axis = plt.subplots(figsize=(7, 4.4))
    axis.plot(dense, magnitude, label="Continuous kernel magnitude")
    axis.scatter(grid.points, samples, s=16, label=f"Original grid ({GRID_BITS} bits)")
    axis.set(
        title="LCHS kernel",
        xlabel="Signed grid coordinate k",
        ylabel="Unnormalized magnitude",
    )
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
