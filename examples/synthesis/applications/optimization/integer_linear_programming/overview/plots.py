"""Plot the actual integer optimization trace, observations and separate reference."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .workflow import Workflow, model_row


def optimization_figure(workflow: Workflow) -> Figure:
    """Preserve source trace, 30-bin histogram and original-domain solution roles."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot([row.cost for row in workflow.trace])
    axes[0, 0].set(
        xlabel="Actual quantum evaluation",
        ylabel="Retained-cost lower-tail CVaR 0.7",
        title=f"Native bounded search; converged={workflow.optimizer.converged}",
    )
    rows = (*workflow.samples, *workflow.uniform)
    bins = np.linspace(
        min(row.retained_quantum_cost for row in rows),
        max(row.retained_quantum_cost for row in rows) + 1,
        31,
    )
    for label, samples in (
        ("Optimized", workflow.samples),
        ("Uniform", workflow.uniform),
    ):
        total = sum(row.count for row in samples)
        axes[0, 1].hist(
            [row.retained_quantum_cost for row in samples],
            bins=bins,
            weights=[row.count / total for row in samples],
            alpha=0.6,
            label=label,
        )
    axes[0, 1].legend()
    axes[0, 1].set(
        xlabel="Retained quantum penalty cost",
        ylabel="Empirical probability",
        title="Actual optimized and uniform observations; common bins",
    )
    reference = model_row((*workflow.exact_reference, 0, 0))
    for axis, row, title in (
        (
            axes[1, 0],
            workflow.best_feasible,
            "Best actually observed feasible assignment",
        ),
        (axes[1, 1], reference, "Exact bounded classical reference"),
    ):
        if row is None:
            axis.axis("off")
            axis.text(0.1, 0.5, "No feasible assignment was observed")
            continue
        axis.bar(("x0", "x1", "x2"), row.assignment[:3], color="cornflowerblue")
        axis.set(ylim=(0, 3.5), ylabel="Original bounded integer value", title=title)
        axis.text(
            0.05,
            0.9,
            f"x={row.assignment[:3]}; objective={row.objective}\n"
            f"A*x={row.lhs} <= b=(1, 2, 3)\nFeasible={row.feasible}",
            transform=axis.transAxes,
            va="top",
        )
    optimized = sum(row.count for row in workflow.samples if row.feasible)
    uniform = sum(row.count for row in workflow.uniform if row.feasible)
    figure.suptitle(
        "Original ILP, all three inequalities; actual feasible observations: "
        f"{optimized}/{sum(row.count for row in workflow.samples)} optimized, "
        f"{uniform}/{sum(row.count for row in workflow.uniform)} uniform"
    )
    return figure
