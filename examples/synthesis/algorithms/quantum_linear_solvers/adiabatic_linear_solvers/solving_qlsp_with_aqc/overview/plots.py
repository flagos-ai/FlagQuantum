"""Plot fresh spectra and actual native and direct-solve distributions."""

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .model import LinearSystem, spectrum
from .workflow import EvolutionResult


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def save_spectrum(model: LinearSystem, path: Path, *, zoom: bool = False) -> Path:
    """Plot all eight eigenvalues, with the original zoom as an option."""
    pyplot = _pyplot()
    points, eigenvalues = spectrum(model)
    figure, axes = pyplot.subplots(figsize=(8, 4.5))
    offsets = np.linspace(-0.05, 0.05, 8)
    for index, values in enumerate(eigenvalues.T):
        axes.plot(points, values + offsets[index], label=f"Eigenvalue {index + 1}")
    if zoom:
        axes.set_ylim(-1.5, 1.5)
    axes.set(
        xlabel="Interpolation fraction",
        ylabel="Eigenvalue",
        title="Adiabatic spectrum with display offsets",
    )
    axes.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path


def save_comparison(
    model: LinearSystem, evolution: EvolutionResult, path: Path
) -> Path:
    """Compare the unconditional eight-point distributions."""
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for axis, state, title in zip(
        axes,
        (evolution.state, model.exact),
        ("Native evolution", "Direct solve"),
        strict=True,
    ):
        axis.bar(range(8), np.abs(state) ** 2, color="#376e9a")
        axis.set(xlabel="Basis word", title=title, xticks=range(8), ylim=(0, 1))
    axes[0].set_ylabel("Probability")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    pyplot.close(figure)
    return path
