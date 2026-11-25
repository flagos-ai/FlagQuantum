"""Convergence figures and explicit source-table columns for actual observations."""

from typing import TYPE_CHECKING

from .model import feasible_value
from .workflow import Optimization

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_convergence(run: Optimization) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(range(1, len(run.evaluations) + 1), [r.cost for r in run.evaluations])
    axes.set_xlabel("Actual objective evaluations")
    axes.set_ylabel("Cost")
    axes.set_title(f"{run.model.capitalize()} cost convergence ({run.termination})")
    figure.tight_layout()
    return figure


def sample_table(run: Optimization) -> tuple[dict[str, int | float], ...]:
    if run.model == "maxcut":
        return tuple(
            dict(word=row.values[0], count=row.count, frequency=row.probability)
            for row in run.table_samples
        )
    return tuple(
        {
            "a": row.values[0],
            "b": row.values[1],
            **({"slack": row.values[2]} if run.model == "penalty" else {}),
            "count": row.count,
            "frequency": row.probability,
            "feasible_value": feasible_value(row.values[0], row.values[1]),
        }
        for row in run.table_samples
    )
