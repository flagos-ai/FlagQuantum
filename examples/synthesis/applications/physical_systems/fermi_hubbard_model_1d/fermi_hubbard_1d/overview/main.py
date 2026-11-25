"""Run the original four-site Fermi–Hubbard comparisons."""

from pathlib import Path

from .plots import (
    amplitudes,
    dynamics,
    free_dynamics,
    frozen_densities,
    initial_densities,
    spread_plot,
)
from .workflow import (
    demo_orbitals,
    initial_one_electron,
    interacting,
    noninteracting,
    vanishing_hopping,
)

DIRECTORY = Path("artifacts/development/fermi-hubbard-1d")


def run_example() -> None:
    demo = demo_orbitals()
    single, reference, fidelity = initial_one_electron()
    interacting_result = interacting()
    free = noninteracting()
    frozen = vanishing_hopping()
    print("Four-mode reconstruction error:", demo.reconstruction_error)
    print("Single-electron fidelity:", fidelity)
    for label, trajectory in (
        ("Interacting", interacting_result),
        ("Free", free),
        ("Vanishing hopping", frozen),
    ):
        print(
            label,
            "time points:",
            len(trajectory.times),
            "resources:",
            trajectory.program.report["resources"],
        )
    print(
        "Frozen density errors:",
        float((frozen.charge - frozen.charge[0]).abs().max()),
        float((frozen.spin - frozen.spin[0]).abs().max()),
    )
    for figure in (
        amplitudes(single, reference, DIRECTORY / "initial-amplitudes.png"),
        initial_densities(interacting_result, DIRECTORY / "initial-densities.png"),
        dynamics(interacting_result, DIRECTORY / "interacting-densities.png"),
        spread_plot(interacting_result, DIRECTORY / "spread.png"),
        free_dynamics(DIRECTORY / "free-analytic.png"),
        free_dynamics(DIRECTORY / "free-trotter.png", free),
        frozen_densities(frozen, DIRECTORY / "frozen-densities.png"),
    ):
        print(figure)


if __name__ == "__main__":
    run_example()
