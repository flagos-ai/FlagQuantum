"""Plot actual optimization samples and the selected packing geometry."""

from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

from .model import HEIGHT, RECTANGLES, WIDTH
from .workflow import SHOTS, PackingRun


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def _save(pyplot: Any, figure: Any, path: Path) -> Path:
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path


def save_trace(result: PackingRun, path: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7, 4))
    axis.plot(
        range(1, len(result.evaluations) + 1),
        [record.mean_cost for record in result.evaluations],
    )
    axis.set(
        xlabel="Actual objective query",
        ylabel="Sampled mean penalized cost",
        title="QAOA rectangle packing",
    )
    return _save(pyplot, figure, path)


def save_costs(result: PackingRun, path: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7, 4))
    frequencies: dict[int, int] = defaultdict(int)
    for word, count in result.sampled:
        frequencies[int(result.model.cost[word])] += count
    axis.hist(
        list(frequencies),
        weights=[count / SHOTS for count in frequencies.values()],
        bins=30,
    )
    axis.set(
        xlabel="Penalized cost",
        ylabel="Sampled probability",
        title="Final sampled costs",
    )
    return _save(pyplot, figure, path)


def save_packing(result: PackingRun, path: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(6, 3.7))
    colors = ("#286c94", "#389a88", "#d17043")
    for i, x, y in result.placements:
        width, height = RECTANGLES[i]
        axis.add_patch(
            pyplot.Rectangle(
                (x, y), width, height, edgecolor="black", facecolor=colors[i], alpha=0.7
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            f"R{i}",
            color="white",
            ha="center",
            va="center",
        )
    axis.set(
        xlim=(0, WIDTH),
        ylim=(HEIGHT, 0),
        xticks=list(range(WIDTH + 1)),
        yticks=list(range(HEIGHT + 1)),
        title=f"Observed packing, cost {float(result.model.cost[result.word]):g}",
        xlabel="x grid site",
        ylabel="y grid site",
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.35)
    return _save(pyplot, figure, path)
