"""Execute the original-scale fresh LiH chemistry and tapered UCCSD VQE."""

import argparse
from pathlib import Path

from .model import (
    ACTIVE_ORBITALS,
    ACTIVE_PARTICLES,
    BASIS,
    FROZEN,
    GEOMETRY,
    WIDTH,
    Molecule,
    regenerate,
)
from .symmetry import Sectors, analyze
from .variational import Optimization, fit


def print_model(model: Molecule) -> None:
    print("LiH geometry and basis:", (GEOMETRY, BASIS))
    print("Spatial orbital energies:", model.orbital_energies)
    print("Fresh molecular method energies:", model.method_energies)
    print(
        "Frozen and active orbitals with particle counts:",
        (FROZEN, ACTIVE_ORBITALS, ACTIVE_PARTICLES),
    )
    print("Canonical fermion terms:", len(model.fermion.terms))
    print(
        "Fresh core-frozen and native FCI energies:",
        (model.active_energy, model.native_active_energy),
    )


def print_sectors(sectors: Sectors) -> None:
    print("Jordan-Wigner term count:", len(sectors.hamiltonian.terms))
    print("Diagonal generators:", tuple(g.terms[0].factors for g in sectors.generators))
    print("Dual X positions:", tuple(x.terms[0].factors[0][0] for x in sectors.dual_x))
    print("Transformed term count:", len(sectors.block.terms))
    print("All sector minima:", sectors.minima)
    print("Largest independent sector spectral error:", sectors.spectral_error)
    print(
        "Hartree-Fock sector and reduced basis word:",
        (sectors.hf_sector, sectors.hf_word),
    )
    print("Orbitals encoded and tapered width:", (WIDTH, sectors.reduction.n_qubits))


def print_optimization(result: Optimization) -> None:
    print("UCCSD excitation term counts:", result.excitation_terms)
    print("Thresholded VQE qubits and terms:", (result.width, result.terms))
    print("Actual VQE queries:", len(result.probes))
    print("Converged and budget exhausted:", (result.convergence, result.budget_stop))
    print("Independent maximum ansatz state error:", result.state_error)
    print("First VQE energy:", result.probes[0].energy)
    print("Last VQE energy:", result.probes[-1].energy)
    print("Best observed VQE energy:", min(probe.energy for probe in result.probes))


def save_trace(model: Molecule, result: Optimization, directory: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    energies = [probe.energy for probe in result.probes]
    index = min(range(len(energies)), key=energies.__getitem__)
    axis.plot(
        range(1, len(energies) + 1),
        energies,
        color="#225980",
        linewidth=1.2,
        label="Evaluated UCCSD energy",
    )
    axis.scatter(
        index + 1,
        energies[index],
        color="#bf562d",
        zorder=4,
        label="Best observed query",
    )
    axis.axhline(
        model.method_energies[-1][1],
        linestyle="--",
        color="#676d79",
        label="Fresh full-space FCI",
    )
    axis.set(
        xlabel="Objective evaluation",
        ylabel="Energy (Ha)",
        title="LiH tapered UCCSD optimization",
    )
    axis.legend(loc="upper right", frameon=False)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / "lih-vqe-trace.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("lih-figures"))
    options = parser.parse_args()
    model = regenerate()
    print_model(model)
    sectors = analyze(model.fermion)
    print_sectors(sectors)
    result = fit(sectors)
    print_optimization(result)
    save_trace(model, result, options.figures)


if __name__ == "__main__":
    main()
