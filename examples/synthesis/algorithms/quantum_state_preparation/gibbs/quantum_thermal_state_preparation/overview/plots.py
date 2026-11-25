"""Plot the complete frequency windows and finite Gibbs observations."""

from importlib import import_module
from pathlib import Path
from typing import Any

from .model import (
    ENERGIES,
    SPECTRUM_BITS,
    THERMAL_SHOTS,
    fourier_spacing,
    frequency,
    gaussian,
)
from .workflow import Workflow


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def _save(pyplot: Any, figure: Any, output: Path) -> Path:
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    pyplot.close(figure)
    return output


def save_window(output: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(6.7, 3.8))
    amplitudes = gaussian(SPECTRUM_BITS)
    grid = [-2 + 4 * i / (len(amplitudes) - 1) for i in range(len(amplitudes))]
    axis.scatter(grid, amplitudes, s=16)
    axis.set(
        title="Six-qubit Gaussian window",
        xlabel="Window coordinate",
        ylabel="Amplitude",
    )
    return _save(pyplot, figure, output)


def save_spectra(result: Workflow, output: Path) -> Path:
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(9, 4), sharey=True)
    bits = SPECTRUM_BITS
    scale = fourier_spacing(bits)
    ticks = [frequency(word, bits) * scale for word in range(1 << bits)]
    for axis, record, title in zip(
        axes,
        (result.gaussian, result.uniform),
        ("Gaussian window", "Uniform window"),
        strict=True,
    ):
        axis.scatter(ticks, record.counts, s=12)
        axis.set(title=title, xlabel="Energy difference", xlim=(-5, 5))
    axes[0].set_ylabel("Actual counts")
    return _save(pyplot, figure, output)


def save_gibbs(result: Workflow, output: Path) -> Path:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(6.5, 4))
    words = list(range(4))
    axis.bar(
        [word - 0.18 for word in words],
        [count / THERMAL_SHOTS for count in result.thermal.counts],
        0.36,
        label="Measured",
    )
    axis.bar(
        [word + 0.18 for word in words],
        result.exact_gibbs,
        0.36,
        label="Gibbs prediction",
    )
    axis.set(
        title="Thermal state after 50 reset cycles",
        xlabel="System word and energy",
        ylabel="Probability",
        xticks=words,
        xticklabels=[f"{word:02b} ({energy})" for word, energy in enumerate(ENERGIES)],
    )
    axis.legend()
    return _save(pyplot, figure, output)
