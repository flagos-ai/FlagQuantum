"""Compare freshly measured local resources and the labeled archive."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

from .archive import QISKIT, Record

Metric = Literal["cx", "seconds"]


def save(
    series: tuple[tuple[Record, ...], tuple[Record, ...]],
    metric: Metric,
    target: Path,
    *,
    historical: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, (axes, widths) = plt.subplots(2, 1, figsize=(8, 6.8), sharex=True)
    try:
        heading = "Classiq (source CSV)" if historical else "FlagQuantum"
        for label, rows, color, marker in (
            ("Qiskit (source record)", QISKIT, "#786099", "s"),
            (f"{heading}, width first", series[0], "#25699c", "D"),
            (f"{heading}, CX first", series[1], "#c06a27", "o"),
        ):
            x = tuple(row.position_bits for row in rows)
            values = tuple(row.cx if metric == "cx" else row.seconds for row in rows)
            axes.plot(x, values, color=color, marker=marker, markersize=4, label=label)
            widths.plot(
                x,
                tuple(row.width for row in rows),
                color=color,
                marker=marker,
                markersize=4,
            )
        axes.set_yscale("log")
        axes.set_ylabel("CX count" if metric == "cx" else "Generation time (s)")
        axes.set_title(
            "Published circle-walk results"
            if historical
            else "Local circle-walk generation"
        )
        axes.legend()
        widths.set_ylabel("Physical wires")
        widths.set_xlabel("Position bits, log₂(circle size)")
        for axis in (axes, widths):
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(target, dpi=145)
    finally:
        plt.close(fig)
