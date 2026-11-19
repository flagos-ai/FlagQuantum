"""Compare signed amplitudes for all four structures on a common figure."""

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from .workflow import Projection


def save_projections(results: Sequence[Projection], output: Path) -> Path:
    """Show the native and independent real vectors before rescaling."""
    pyplot = import_module("matplotlib.pyplot")
    figure, axes = pyplot.subplots(2, 2, figsize=(9, 6))
    for axis, result in zip(axes.flat, results, strict=True):
        words = range(len(result.actual))
        axis.scatter(words, result.expected.real.tolist(), label="Matrix product")
        axis.scatter(
            words, result.actual.real.tolist(), marker="x", label="Native projection"
        )
        axis.set(
            title=result.structure.name.title(),
            xlabel="Data word",
            ylabel="Signed amplitude",
            xticks=list(words),
        )
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    pyplot.close(figure)
    return output
