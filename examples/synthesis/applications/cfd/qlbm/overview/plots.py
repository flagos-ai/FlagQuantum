"""Plot the source CFL event table, sampled marginals, and reflecting geometry."""

from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from .model import GRID, MAGNITUDES, X_LIMITS, Y_LIMITS, schedule


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def _save(pyplot: Any, figure: Any, path: Path) -> Path:
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path


def save_schedule(path: Path) -> Path:
    """Draw all nine events and four magnitude classes."""
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(5.4, 4.5))
    events = schedule()
    table = [[int(speed in active) for speed in MAGNITUDES] for _, active in events]
    image = axis.imshow(table, origin="lower", aspect="auto", vmin=0, vmax=1)
    axis.set(
        title="CFL streaming schedule",
        xlabel="Velocity magnitude",
        ylabel="Event time",
        xticks=list(MAGNITUDES),
        yticks=list(range(len(events))),
        yticklabels=[str(t) for t, _ in events],
    )
    figure.colorbar(image, ax=axis, ticks=(0, 1), label="Streams at event")
    return _save(pyplot, figure, path)


def save_marginal(values: torch.Tensor, quantity: str, path: Path) -> Path:
    """Draw a measured joint marginal with named horizontal and vertical axes."""
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(5, 4.5))
    image = axis.imshow(values.tolist(), origin="lower", vmin=0)
    ticks = list(range(values.shape[0]))
    axis.set(
        title=f"Sampled {quantity} distribution",
        xlabel=f"x {quantity}",
        ylabel=f"y {quantity}",
        xticks=ticks,
        yticks=ticks,
    )
    figure.colorbar(image, ax=axis, label="Probability")
    return _save(pyplot, figure, path)


def save_reflection(path: Path) -> Path:
    """Illustrate the two boundary tests on the source obstacle grid."""
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(5, 5))
    points = [(x, y) for x in range(GRID) for y in range(GRID)]
    xs, ys = zip(*points, strict=True)
    axis.scatter(xs, ys, marker="+", color="#52656e", label="Lattice site")
    left, right = X_LIMITS
    bottom, top = Y_LIMITS
    patch = pyplot.Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        alpha=0.26,
        color="#d47d49",
        label="Reflecting obstacle",
    )
    axis.add_patch(patch)
    for (start, stop), color in (
        (((0.7, 3), (left, 3)), "#2a6681"),
        (((left, 3.2), (0.7, 3.2)), "#ba4a2d"),
        (((3.5, 6.6), (3.5, top)), "#2a6681"),
        (((3.7, top), (3.7, 6.6)), "#ba4a2d"),
    ):
        axis.annotate(
            "",
            xy=stop,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2),
        )
    axis.plot([], [], color="#2a6681", label="Incident")
    axis.plot([], [], color="#ba4a2d", label="Reflected")
    axis.legend(fontsize=8, loc="upper right")
    axis.set(
        title="Axis-wise reflection after streaming",
        xlim=(-0.5, 7.5),
        ylim=(-0.5, 7.5),
        xlabel="x lattice site",
        ylabel="y lattice site",
        xticks=list(range(GRID)),
        yticks=list(range(GRID)),
    )
    axis.set_aspect("equal")
    return _save(pyplot, figure, path)
