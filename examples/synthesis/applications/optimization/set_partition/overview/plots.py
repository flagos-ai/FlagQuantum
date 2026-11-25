"""Plot actual costs and partitions with the exact reference labeled separately."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .workflow import Workflow, costs, partitions


def partition_figure(workflow: Workflow) -> Figure:
    """Retain the cost trace, sampled histogram and observed/reference group roles."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot([entry.cost for entry in workflow.trace])
    axes[0, 0].set(
        xlabel="Actual quantum evaluation",
        ylabel="Lower-tail CVaR 0.7",
        title=f"Bounded native optimization; converged={workflow.optimizer.converged}",
    )
    maximum = max(row.cost for row in (*workflow.samples, *workflow.uniform))
    bins = np.linspace(0, maximum + 1, 51)
    for label, samples in (
        ("Optimized", workflow.samples),
        ("Uniform", workflow.uniform),
    ):
        total = sum(row.count for row in samples)
        axes[0, 1].hist(
            [row.cost for row in samples],
            bins=bins,
            weights=[row.count / total for row in samples],
            alpha=0.6,
            label=label,
        )
    axes[0, 1].legend()
    axes[0, 1].set(
        xlabel="Squared partition difference",
        ylabel="Empirical probability",
        title="Actual seeded samples; common histogram bins",
    )
    for axis, word, title in (
        (axes[1, 0], workflow.best_observed.word, "Best actually observed partition"),
        (
            axes[1, 1],
            workflow.exact_reference_word,
            "Exact bounded classical reference",
        ),
    ):
        selected, other = partitions(word, workflow.weights)
        axis.bar(
            ["Selected (bit 1)", "Other (bit 0)"],
            [sum(selected), sum(other)],
            color=["cornflowerblue", "tomato"],
        )
        axis.set(ylabel="Group sum", title=title)
        axis.text(
            0.5,
            0.9,
            f"Selected={selected}\nOther={other}\nSquared difference={costs(workflow.weights)[word]}",
            ha="center",
            va="top",
            transform=axis.transAxes,
        )
    figure.suptitle(
        f"Source Notebook weights {workflow.weights}; total={sum(workflow.weights)}"
    )
    return figure
