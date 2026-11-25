"""Run the Drude-oscillator workflow from the command line."""

from pathlib import Path

import numpy as np

from .model import (
    POLARIZABILITY,
    finite_ground_energies,
    harmonic_ground_energy,
    pairwise_interaction,
)
from .plots import save_energy_curves, save_trace
from .workflow import VariationalRun, optimize_distance, sweep

OUTPUT = Path("artifacts/development/drude-oscillator")
DIMENSIONS = (2, 4, 8, 16)


def show_run(run: VariationalRun) -> None:
    """Report the actual final synthesized energy and separate finite reference."""
    print(
        f"{run.oscillator_count} oscillators, R={run.distance:.2f}: "
        f"iterations={run.iterations}, evaluations={run.evaluations}, "
        f"VQE={run.energy:.12f}, "
        f"finite reference={run.reference_energy:.12f}"
    )


def main() -> None:
    test = optimize_distance(2, 3.0)
    show_run(test)
    save_trace(test, OUTPUT / "vqe-trace.png")
    two = sweep(2)
    for run in two:
        show_run(run)
    distances = np.arange(3.1, 6.1, 0.2)
    harmonic = (
        (
            "Harmonic solution",
            harmonic_ground_energy(-4 * POLARIZABILITY / distances**3),
        ),
    )
    save_energy_curves(
        distances,
        harmonic,
        two,
        xlabel="Distance (Å)",
        reference_energy=2.0,
        title="Two-oscillator London dispersion",
        path=OUTPUT / "two-oscillators.png",
    )
    truncation = tuple(
        (f"d = {dimension}", finite_ground_energies(distances, dimension))
        for dimension in DIMENSIONS
    )
    save_energy_curves(
        distances,
        (*truncation, *harmonic),
        (),
        xlabel="Distance (Å)",
        reference_energy=2.0,
        title="Finite-basis truncation by distance",
        path=OUTPUT / "finite-distance.png",
    )
    coupling = -np.linspace(0.01, 2.0, 15)
    coupling_distances = (-4 * POLARIZABILITY / coupling) ** (1 / 3)
    coupling_curves = tuple(
        (f"d = {dimension}", finite_ground_energies(coupling_distances, dimension))
        for dimension in DIMENSIONS
    )
    save_energy_curves(
        -coupling,
        (*coupling_curves, ("Harmonic solution", harmonic_ground_energy(coupling))),
        (),
        xlabel="Negative coupling",
        reference_energy=0.0,
        title="Finite-basis truncation by coupling",
        path=OUTPUT / "finite-coupling.png",
    )
    three = sweep(3)
    for run in three:
        show_run(run)
    save_energy_curves(
        distances,
        (
            ("Pair interactions", 3 + pairwise_interaction(distances)),
            (
                "Finite three-body diagonalization",
                finite_ground_energies(distances, 4, 3),
            ),
        ),
        three,
        xlabel="Distance (Å)",
        reference_energy=3.0,
        title="Three-oscillator London dispersion",
        path=OUTPUT / "three-oscillators.png",
    )


if __name__ == "__main__":
    main()
