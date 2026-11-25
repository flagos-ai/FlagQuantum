"""Plot actual Bayesian posteriors for each original experiment policy."""

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from .workflow import THRESHOLD, Update


def save_posterior(history: Sequence[Update], title: str, output: Path) -> Path:
    """Draw the four posterior curves over independently sampled observations."""
    pyplot = import_module("matplotlib.pyplot")
    figure, axis = pyplot.subplots(figsize=(7.8, 4.3))
    iterations = range(1, len(history) + 1)
    for index, label in enumerate(("X", "2X", "Z", "2Z")):
        axis.plot(
            iterations,
            [record.posterior[index] for record in history],
            "o-",
            label=label,
            markersize=4,
        )
    axis.axhline(THRESHOLD, linestyle="--", color="#686868", linewidth=1)
    axis.set(
        title=title,
        xlabel="Actual measurements",
        ylabel="Posterior probability",
        ylim=(-0.035, 1.035),
        xticks=list(iterations),
    )
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    pyplot.close(figure)
    return output
