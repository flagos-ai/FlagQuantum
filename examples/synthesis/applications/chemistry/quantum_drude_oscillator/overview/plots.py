"""Show the source energy sweeps, truncation error, and three-body interaction."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from .model import HALF_HBAR_OMEGA, RealArray
from .workflow import VariationalRun


def save_trace(run: VariationalRun, path: Path) -> Path:
    """Plot every objective evaluation at the initial test separation."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        range(1, run.evaluations + 1), run.objective_energies, label="Native evaluation"
    )
    axis.axhline(
        run.reference_energy,
        linestyle="--",
        color="#767676",
        label="Finite ground energy",
    )
    axis.set(
        xlabel="Actual objective evaluation",
        ylabel="Energy (half ℏω)",
        title="Two oscillators at R = 3 Å",
    )
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def save_energy_curves(
    domain: RealArray,
    curves: Sequence[tuple[str, RealArray]],
    observations: Sequence[VariationalRun],
    *,
    xlabel: str,
    reference_energy: float,
    title: str,
    path: Path,
) -> Path:
    """Show each regenerated reference curve and actual VQE observations."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    scale = HALF_HBAR_OMEGA if reference_energy else 1.0
    for label, energies in curves:
        axis.plot(domain, (energies - reference_energy) * scale, label=label)
    if observations:
        axis.plot(
            [run.distance for run in observations],
            [(run.energy - reference_energy) * scale for run in observations],
            "s",
            label="Native VQE",
        )
    if reference_energy:
        axis.axhline(
            0, color="#767676", linestyle="--", linewidth=1, label="Infinite separation"
        )
    axis.set(
        xlabel=xlabel,
        ylabel=(
            "Interaction energy (eV)" if reference_energy else "Ground energy (half ℏω)"
        ),
        title=title,
    )
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
