"""Run the original orbital and two-mode pairing cases."""

from pathlib import Path

from .plots import plot_slater
from .workflow import demo, run_pairing, run_slater


def run_example() -> None:
    energies, _, error = demo()
    print("Demonstration spectrum:", energies.tolist())
    print("Orbital reconstruction error:", error)
    slater = run_slater()
    paired = run_pairing()
    print("Slater fidelity:", slater.fidelity)
    print("Canonical positive energies:", paired.decomposition.energies.tolist())
    print("Full Fock spectrum:", paired.eigenvalues.tolist())
    print("Paired fidelity:", paired.fidelity)
    print("Vacuum annihilator residual:", paired.vacuum_residual)
    print("Slater resources:", slater.program.report["resources"])
    print("Pairing resources:", paired.program.report["resources"])
    print(
        plot_slater(
            slater, Path("artifacts/development/fermionic-gaussian/orbital.png")
        )
    )


if __name__ == "__main__":
    run_example()
