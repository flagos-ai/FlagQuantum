"""Plot actual optimizer evaluations and observed source-cost probability mass."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Workflow


def plot_results(workflow: Workflow, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / "cost-convergence.png"
    figure, axes = plt.subplots()
    axes.plot([value.cost for value in workflow.trace])
    axes.set_xlabel("Objective evaluation")
    axes.set_ylabel("Exact mean penalized cost")
    axes.set_title("Antenna placement cost convergence")
    figure.tight_layout()
    figure.savefig(trace)
    plt.close(figure)
    histogram = directory / "sampled-costs.png"
    figure, axes = plt.subplots()
    shots = sum(row.count for row in workflow.samples)
    axes.hist(
        [row.cost for row in workflow.samples],
        weights=[row.count / shots for row in workflow.samples],
        bins=10,
        edgecolor="black",
    )
    axes.set_xlabel("Joint penalized cost")
    axes.set_ylabel("Observed probability")
    axes.set_title("Actual3000-shot antenna observations")
    figure.tight_layout()
    figure.savefig(histogram)
    plt.close(figure)
    return trace, histogram
