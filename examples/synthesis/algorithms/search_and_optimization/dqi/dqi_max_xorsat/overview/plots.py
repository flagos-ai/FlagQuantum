"""Plot the source graph and all four Max-XORSAT score distributions."""

import math
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any

from .model import DQI_SHOTS, EDGES
from .workflow import Workflow

CYCLE = (3, 4, 1, 5, 0, 2)


def _save(pyplot: Any, figure: Any, path: Path) -> Path:
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path


def save_graph(output: Path) -> Path:
    pyplot = import_module("matplotlib.pyplot")
    figure, axis = pyplot.subplots(figsize=(5, 4))
    positions = {
        node: (math.cos(math.tau * i / 6), math.sin(math.tau * i / 6))
        for i, node in enumerate(CYCLE)
    }
    for left, right in EDGES:
        x, y = positions[left]
        u, v = positions[right]
        axis.plot([x, u], [y, v], color="#375361", linewidth=2)
    for node, (x, y) in positions.items():
        axis.scatter(x, y, s=400, color="#226980", zorder=3)
        axis.text(x, y, str(node), color="white", ha="center", va="center", zorder=4)
    axis.set(
        title="Six-cycle Max-XORSAT instance", xlim=(-1.25, 1.25), ylim=(-1.15, 1.15)
    )
    axis.set_aspect("equal")
    axis.axis("off")
    return _save(pyplot, figure, output)


def save_scores(result: Workflow, output: Path) -> Path:
    pyplot = import_module("matplotlib.pyplot")
    figure, axis = pyplot.subplots(figsize=(7.7, 4.5))
    counts = Counter(result.model.objective)
    observed: Counter[int] = Counter()
    for (_, word), count in result.dqi:
        observed[result.model.objective[word]] += count
    scores = sorted(counts)
    weighted_abs = sum(counts[score] * abs(score) for score in scores)
    weighted_square = sum(counts[score] * score**2 for score in scores)
    series = (
        ("Native DQI", [observed[score] / DQI_SHOTS for score in scores]),
        ("Uniform", [counts[score] / 64 for score in scores]),
        (
            "Absolute-score weighting",
            [counts[score] * abs(score) / weighted_abs for score in scores],
        ),
        (
            "Squared-score weighting",
            [counts[score] * score**2 / weighted_square for score in scores],
        ),
    )
    for index, (label, probabilities) in enumerate(series):
        axis.bar(
            [score + 0.34 * (index - 1.5) for score in scores],
            probabilities,
            0.34,
            label=label,
        )
    axis.set(
        title="Sampled Max-XORSAT scores",
        xlabel="Signed clause score",
        ylabel="Probability",
        xticks=scores,
    )
    axis.legend(fontsize=8)
    return _save(pyplot, figure, output)
