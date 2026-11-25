"""Render the executed circle probabilities and same-basis CX counts."""

from pathlib import Path

import matplotlib.pyplot as plt

from . import sweep, walks


def circle(
    data: walks.Observation,
    directory: Path,
    *,
    classical: tuple[int, ...] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.1))
    try:
        if classical is not None:
            axis.plot(
                data.positions,
                tuple(count / walks.SHOTS for count in classical),
                label="Classical, 10,000 trials",
                color="#b76025",
                linewidth=1.4,
            )
        # Every retained 50-step position has even parity.
        allowed = [
            index for index, position in enumerate(data.positions) if position % 2 == 0
        ]
        axis.plot(
            [data.positions[index] for index in allowed],
            [data.counts[index] / walks.SHOTS for index in allowed],
            label="Quantum, 10,000 draws",
            color="#265e9c",
            marker=".",
            linewidth=1.4,
        )
        axis.set_title(
            "Circle walk from zero"
            if classical is not None
            else "Circle walk with a balanced coin"
        )
        axis.set_xlabel("Signed position")
        axis.set_ylabel("Observed probability")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        name = "circle-samples.png" if classical is not None else "balanced-circle.png"
        figure.savefig(directory / name, dpi=140)
    finally:
        plt.close(figure)


def width_sweep(rows: tuple[sweep.Point, ...], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.1))
    try:
        widths = tuple(row.width_limit for row in rows)
        axis.plot(
            widths,
            tuple(sweep.cost(row.arithmetic)[0] for row in rows),
            label="Built-in increment",
            color="#285f9a",
            marker="s",
            markersize=8,
            markerfacecolor="none",
        )
        axis.plot(
            widths,
            tuple(sweep.cost(row.cascade)[0] for row in rows),
            label="MCX cascade",
            color="#cf6d22",
            marker=".",
            linestyle="--",
        )
        axis.set_yscale("log")
        axis.set_xticks(widths)
        axis.grid(alpha=0.25)
        axis.set_title("1024-site circle, one quantum step")
        axis.set_xlabel("Physical width constraint")
        axis.set_ylabel("CX count in CX/U3 basis")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / "cx-width-sweep.png", dpi=140)
    finally:
        plt.close(figure)
