"""Plot the actual routed sample, exact route, and finite objective trace."""

from importlib import import_module
from itertools import pairwise
from pathlib import Path
from typing import Any

from .model import EDGES
from .workflow import QUANTILE, Workflow

POSITIONS = ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (2.0, 1.0))


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def _save(pyplot: Any, figure: Any, path: Path) -> Path:
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path


def save_graph(
    path: Path,
    *,
    route: tuple[int, ...] | None = None,
    title: str = "Four-city directed graph",
) -> Path:
    """Show every original directed edge and the given feasible route."""
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7.4, 4))
    active = set(pairwise(route)) if route is not None else set()
    for left, right, weight in EDGES:
        x, y = POSITIONS[left]
        u, v = POSITIONS[right]
        is_reverse_pair = {left, right} == {1, 2}
        curvature = 0.16 if is_reverse_pair else 0
        color = "#a83d33" if (left, right) in active else "#4a6879"
        axis.annotate(
            "",
            xy=(u, v),
            xytext=(x, y),
            arrowprops=dict(
                arrowstyle="->",
                shrinkA=16,
                shrinkB=16,
                linewidth=2.5,
                color=color,
                connectionstyle=f"arc3,rad={curvature}",
            ),
        )
        label_x = (
            (x + u) / 2 + (-0.16 if left == 1 else 0.16)
            if is_reverse_pair
            else (x + u) / 2
        )
        axis.text(
            label_x,
            (y + v) / 2 + (0 if is_reverse_pair else 0.08),
            str(weight),
            color=color,
            backgroundcolor="white",
            ha="center",
            va="center",
        )
    available = {(left, right) for left, right, _ in EDGES}
    for left, right in sorted(active - available):
        axis.annotate(
            "",
            xy=POSITIONS[right],
            xytext=POSITIONS[left],
            arrowprops=dict(
                arrowstyle="->",
                shrinkA=16,
                shrinkB=16,
                linestyle="--",
                linewidth=2.5,
                color="#a83d33",
            ),
        )
    for city, (x, y) in enumerate(POSITIONS):
        axis.scatter([x], [y], color="#246783", s=570)
        axis.text(x, y, str(city), color="white", ha="center", va="center")
    axis.set(xlim=(-0.35, 2.35), ylim=(-0.26, 1.32), title=title)
    axis.set_aspect("equal")
    axis.axis("off")
    return _save(pyplot, figure, path)


def save_trace(result: Workflow, path: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7.5, 4))
    axis.plot(
        range(1, len(result.evaluations) + 1), [row.value for row in result.evaluations]
    )
    axis.set(
        title="Finite-sample route optimization",
        xlabel="Actual objective query",
        ylabel=f"Lower {QUANTILE:.0%} mean cost",
    )
    return _save(pyplot, figure, path)


def save_costs(result: Workflow, path: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7.5, 4))
    optimized = result.model.cost[result.optimized_draw].tolist()
    uniform = result.model.cost[result.uniform_draw].tolist()
    shared_range = (
        min(min(optimized), min(uniform)),
        max(max(optimized), max(uniform)),
    )
    for values, label in ((optimized, "Optimized"), (uniform, "Uniform")):
        axis.hist(
            values,
            bins=50,
            range=shared_range,
            alpha=0.5,
            label=label,
            weights=[1 / len(values)] * len(values),
        )
    axis.set(
        title="Final observed cost distributions",
        xlabel="Penalized path cost",
        ylabel="Sampled probability",
    )
    axis.legend()
    return _save(pyplot, figure, path)
