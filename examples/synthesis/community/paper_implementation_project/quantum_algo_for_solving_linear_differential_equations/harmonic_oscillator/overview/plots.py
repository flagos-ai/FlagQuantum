"""Plot analytic and postselected oscillator trajectories and energies."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Observation


def save_comparison(rows: tuple[Observation, ...], path: Path) -> Path:
    """Keep the signed zero crossing and both energy curves visible."""
    times = [row.time for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4))
    left.plot(
        times,
        [row.exact_position for row in rows],
        color="tab:blue",
        label="Analytic position",
    )
    left.plot(
        times,
        [row.exact_velocity for row in rows],
        color="tab:orange",
        label="Analytic velocity",
    )
    left.scatter(
        times,
        [row.position for row in rows],
        marker="o",
        facecolors="none",
        edgecolors="tab:blue",
        label="Quantum position",
    )
    left.scatter(
        times,
        [row.velocity for row in rows],
        marker="s",
        facecolors="none",
        edgecolors="tab:orange",
        label="Quantum velocity",
    )
    left.axhline(0, linewidth=0.7, color="#777777")
    left.set(
        xlabel="Time", ylabel="Signed solution component", title="Harmonic oscillator"
    )
    right.plot(
        times,
        [row.exact_position**2 / 2 for row in rows],
        color="tab:blue",
        label="Analytic potential",
    )
    right.plot(
        times,
        [row.exact_velocity**2 / 2 for row in rows],
        color="tab:orange",
        label="Analytic kinetic",
    )
    right.scatter(
        times,
        [row.potential for row in rows],
        marker="o",
        facecolors="none",
        edgecolors="tab:blue",
        label="Quantum potential",
    )
    right.scatter(
        times,
        [row.kinetic for row in rows],
        marker="s",
        facecolors="none",
        edgecolors="tab:orange",
        label="Quantum kinetic",
    )
    right.set(xlabel="Time", ylabel="Energy", title="Energy comparison")
    for axes in (left, right):
        axes.legend(fontsize=8)
        axes.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
