"""Plot every classical, amplitude-error, fidelity and CX comparison."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Result

LABELS = (
    "fdm-times",
    "fdm",
    "matrix-comparison",
    "matrix-error",
    "proposed",
    "proposed-error",
    "trotter",
    "trotter-error",
    "custom",
    "custom-error",
    "all-errors",
    "qubit-cx",
    "step-cx",
)
Series = tuple[Sequence[float], Sequence[float], str]


def save(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    def render(
        label: str, title: str, xlabel: str, ylabel: str, series: tuple[Series, ...]
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.6, 5), layout="constrained")
        for x, y, name in series:
            ax.plot(x, y, label=name)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=9)
        fig.savefig(directory / (label + ".png"), dpi=140)
        plt.close(fig)

    x = result.axis
    scale = result.norm
    baseline = tuple(v * scale for v in result.reference)
    exact = tuple(v * scale for v in result.exponential)
    render(
        "fdm-times",
        "Original RK4 profiles on the source grid",
        "Position",
        "Profile",
        tuple(
            (x, profile, f"Requested t = {t:g}")
            for t, profile in zip(
                result.classical_times, result.classical_profiles, strict=True
            )
        ),
    )
    render(
        "fdm",
        "Source t = 1 readout at RK4 index 26 (t = 0.988)",
        "Position",
        "Normalized amplitude",
        ((x, result.reference, "RK4 reference"),),
    )
    render(
        "matrix-comparison",
        "Ten exact matrix-exponential steps and the RK4 readout",
        "Position",
        "Profile",
        ((x, baseline, "RK4 reference"), (x, exact, "Matrix exponential")),
    )
    matrix_error = tuple(
        abs(a - b) for a, b in zip(result.reference, result.exponential, strict=True)
    )
    render(
        "matrix-error",
        "Matrix-exponential difference from the RK4 readout",
        "Position",
        "Absolute amplitude difference",
        ((x, matrix_error, "Matrix exponential"),),
    )
    errors = [(x, matrix_error, "Matrix exponential")]
    for curve in result.curves:
        profile = tuple(v * scale for v in curve.corrected)
        error = tuple(
            abs(a - b) for a, b in zip(result.reference, curve.corrected, strict=True)
        )
        render(
            curve.name,
            f"{curve.name.title()} • 10 steps • fidelity {100*curve.fidelity:.2f}%",
            "Position",
            "Profile",
            ((x, baseline, "RK4 reference"), (x, profile, curve.name)),
        )
        render(
            curve.name + "-error",
            f"{curve.name.title()} amplitude difference",
            "Position",
            "Absolute amplitude difference",
            ((x, error, curve.name),),
        )
        errors.append((x, error, curve.name))
    render(
        "all-errors",
        "All original amplitude-error comparisons",
        "Position",
        "Absolute amplitude difference",
        tuple(errors),
    )
    variants = ("proposed", "custom", "trotter")
    resource_series = tuple(
        (result.qubit_widths, counts, label)
        for label, counts in zip(variants, result.qubit_cx, strict=True)
    )
    render(
        "qubit-cx",
        "Original three-to-six-qubit compilation sweep",
        "Qubits",
        "CX count",
        (
            *resource_series,
            (result.qubit_widths, result.qubit_theory, "Paper estimate"),
        ),
    )
    render(
        "step-cx",
        "Original three-qubit step-count sweep",
        "Evolution steps",
        "CX count",
        tuple(
            (result.step_counts, counts, label)
            for label, counts in zip(variants, result.step_cx, strict=True)
        ),
    )
