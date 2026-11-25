"""Source Ising convergence and finite weighted energy histograms."""

from typing import TYPE_CHECKING

from .sampling import Sample
from .workflow import Optimization

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_results(
    result: Optimization, optimized: tuple[Sample, ...], uniform: tuple[Sample, ...]
) -> tuple["Figure", "Figure"]:
    import matplotlib.pyplot as plt

    convergence, axis = plt.subplots(layout="constrained")
    axis.plot(range(1, len(result.trace) + 1), result.trace)
    axis.set_xlabel("Objective evaluations")
    axis.set_ylabel("Lower-tail cost (quantile 0.7)")
    axis.set_title("Cost convergence; 100-evaluation budget")
    histogram, axis = plt.subplots(layout="constrained")
    energies = [row.energy for row in uniform]
    lower, upper = min(energies), max(energies)
    bins = [lower + i * ((upper - lower) / 30) for i in range(31)]
    for label, rows in (("optimized", optimized), ("uniform", uniform)):
        axis.hist(
            [row.energy for row in rows],
            bins=bins,
            weights=[row.probability for row in rows],
            edgecolor="black",
            alpha=0.6,
            label=label,
        )
    axis.legend()
    axis.set_ylabel("Probability", fontsize=16)
    axis.set_xlabel("Source Ising energy", fontsize=16)
    plt.show()
    return convergence, histogram
