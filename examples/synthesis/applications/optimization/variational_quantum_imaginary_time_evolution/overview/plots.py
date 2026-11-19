"""Plot the fresh weighted graph and the actual finite-sample trajectory."""

import math
from importlib import import_module
from pathlib import Path
from typing import Any

from .model import CutModel
from .reference import direct_cut
from .workflow import Run


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def save_graph(model: CutModel, output: Path) -> Path:
    """Draw all six original weighted edges."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.5, 5))
    points = [(math.cos(math.pi * k / 2), math.sin(math.pi * k / 2)) for k in range(4)]
    for u, v, weight in model.edges:
        x, y = points[u]
        tx, ty = points[v]
        axis.plot([x, tx], [y, ty], color="#667588", zorder=1)
        offset_x, offset_y = (0.0, 0.11) if (u, v) == (0, 2) else (0.11, 0.0)
        axis.text(
            (x + tx) / 2 + offset_x,
            (y + ty) / 2 + offset_y,
            f"{weight:.3f}",
            ha="center",
            va="center",
            backgroundcolor="white",
            zorder=2,
        )
    for node, (x, y) in enumerate(points):
        axis.scatter([x], [y], s=500, color="#2a6681", zorder=3)
        axis.text(x, y, str(node), ha="center", va="center", color="white", zorder=4)
    axis.set(
        xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), title="Four-node weighted MaxCut graph"
    )
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    plt.close(figure)
    return output


def save_trace(result: Run, output: Path) -> Path:
    """Plot observed energies and the independent magnitude of the optimal cut."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    steps = [row.iteration for row in result.rounds]
    axis.plot(steps, [row.energy for row in result.rounds], "o-", label="Update energy")
    axis.plot(
        steps,
        [row.partition_energy for row in result.rounds],
        "s-",
        label="Rounded partition energy",
    )
    axis.axhline(
        -max(direct_cut(result.model, word) for word in range(16)),
        ls="--",
        color="#434343",
        label="Exact minimum",
    )
    axis.set(xlabel="Imaginary-time update", ylabel="Negative cut weight", xticks=steps)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    plt.close(figure)
    return output
