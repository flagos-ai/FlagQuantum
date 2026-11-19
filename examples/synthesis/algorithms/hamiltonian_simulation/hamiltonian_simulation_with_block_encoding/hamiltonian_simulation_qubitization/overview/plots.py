"""Plot rescaled complex projections alongside the independent references."""

from importlib import import_module
from pathlib import Path
from typing import Any

from .workflow import Projection


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def save_comparison(projection: Projection, title: str, path: Path) -> Path:
    """Display real and imaginary parts with the declared scale applied."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(8, 3.7), sharey=True)
    values = projection.raw * projection.scale
    for axis, actual, expected, component in zip(
        axes,
        (values.real, values.imag),
        (projection.reference.real, projection.reference.imag),
        ("Real", "Imaginary"),
        strict=True,
    ):
        axis.scatter(range(4), expected.tolist(), label="Dense reference", marker="o")
        axis.scatter(range(4), actual.tolist(), label="Native projection", marker="x")
        axis.set(title=component, xlabel="Data word", xticks=range(4))
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path
