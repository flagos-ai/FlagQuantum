"""Render all eight numerical and circuit-resource comparisons."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Result

LABELS = (
    "initial",
    "classical",
    "quantum",
    "comparison",
    "cx-time",
    "cx-theory",
    "cx-ratio",
    "controlled-ry",
)


def save(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    def plot(
        label: str,
        title: str,
        xlabel: str,
        ylabel: str,
        series: tuple[tuple[Sequence[float], Sequence[float], str], ...],
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for x, y, name in series:
            ax.plot(x, y, label=name)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        fig.savefig(directory / (label + ".png"), dpi=140)
        plt.close(fig)

    initial = (result.axis, result.boundary, "Initial profile")
    classical = (result.axis, result.classical, "Centered-difference profile")
    quantum = (
        result.axis,
        result.quantum_probability_profile,
        "Scaled quantum probability",
    )
    plot("initial", "Dirichlet initial conditions", "Position", "Profile", (initial,))
    plot(
        "classical",
        "Original centered-difference evolution at t = 1",
        "Position",
        "Profile",
        (initial, classical),
    )
    plot(
        "quantum",
        "Quantum advection at t = 1",
        "Position",
        "32 × probability",
        (quantum,),
    )
    plot(
        "comparison",
        "Original classical and quantum readouts",
        "Position",
        "Profile",
        (initial, classical, quantum),
    )
    measured = (result.times, result.measured_cx, "Native CX count")
    theory = (result.times, result.theory_cx, "Paper CX estimate")
    plot(
        "cx-time",
        "Native circuit cost across the original time sweep",
        "Time",
        "CX count",
        (measured,),
    )
    plot(
        "cx-theory",
        "Measured named-basis resources and source theory",
        "Time",
        "CX count",
        (measured, theory),
    )
    ratio = tuple(
        a / b for a, b in zip(result.measured_cx, result.theory_cx, strict=True)
    )
    plot(
        "cx-ratio",
        "Native CX count divided by the source theory",
        "Time",
        "Ratio",
        ((result.times, ratio, "Native / paper"),),
    )
    plot(
        "controlled-ry",
        "Original two-to-seven-bit controlled RY comparison",
        "Total qubits",
        "CX count",
        (
            (result.cry_widths, result.measured_cry, "Native Gray-code compilation"),
            (result.cry_widths, result.theory_cry, "Paper prediction"),
        ),
    )
