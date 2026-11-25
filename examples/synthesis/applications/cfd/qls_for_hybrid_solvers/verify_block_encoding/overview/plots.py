"""Plot both actual projections against raw normalized matrix products."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Projection


def draw(result: Projection, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(
        range(8), result.reference.real, "o", markersize=8, label="Matrix product"
    )
    axes.plot(
        range(8), result.values.real, "x", markersize=7, label="Native projection"
    )
    axes.set(
        title=f"{result.method.capitalize()} nozzle block encoding",
        xlabel="Data index",
        ylabel="Unnormalized real amplitude",
        xticks=range(8),
    )
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
