"""Draw the measured local timings alongside the source's archived values."""

from pathlib import Path

import matplotlib.pyplot as plt

from .benchmark import HISTORICAL_QISKIT, Row


def save(rows: tuple[Row, ...], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, (time_axes, width_axes) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    try:
        time_axes.scatter(
            range(11, 23),
            HISTORICAL_QISKIT,
            marker="s",
            facecolors="none",
            edgecolors="#7a5c9e",
            label="Qiskit (source record)",
        )
        for route, label, color, marker in (
            ("Python qfunc", "FlagQuantum qfunc", "#285f9a", "o"),
            ("Clause Circuit", "FlagQuantum clause Circuit", "#d07b33", "^"),
        ):
            selected = tuple(row for row in rows if row.route == route)
            counts = tuple(row.variables for row in selected)
            time_axes.scatter(
                counts,
                tuple(row.seconds for row in selected),
                c=color,
                marker=marker,
                label=label,
            )
            width_axes.scatter(
                counts,
                tuple(row.resources.n_qubits for row in selected),
                c=color,
                marker=marker,
                label=label,
            )
        time_axes.set_yscale("log")
        time_axes.set_ylabel("Generation time (s)")
        time_axes.set_title("Native 3-SAT oracle generation")
        time_axes.legend(loc="upper right")
        width_axes.set_ylabel("Physical wires")
        width_axes.set_xlabel("Boolean variables")
        for axes in (time_axes, width_axes):
            axes.grid(alpha=0.25)
            axes.set_axisbelow(True)
        figure.tight_layout()
        figure.savefig(target, dpi=150)
    finally:
        plt.close(figure)
