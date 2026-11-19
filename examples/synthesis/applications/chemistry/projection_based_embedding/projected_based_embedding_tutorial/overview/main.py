"""Run the original cc-pVDZ water embedding and thirty-parameter singles VQE."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import workflow

HARTREE_TO_EV = 27.2114079527
ARCHIVED_DISABLED_DIAGONALIZATION = -77.07666711404943


def report(result: workflow.Result) -> None:
    molecule, optimization = result.molecule, result.optimization
    print("Original water: cc-pVDZ, B3LYP, 24 AOs, 10 electrons, oxygen fragment")
    print("Fragment electrons=10, environment electrons=0, active spatial orbitals=8")
    print(f"Fresh DFT energy={molecule.dft_energy}, active FCI={molecule.fci_energy}")
    print(
        f"Density trace={molecule.density_trace}, active metric error={molecule.active_metric_error}"
    )
    print(f"Native canonical fermion terms={len(molecule.fermion.terms)}")
    print(f"Spin parity: 16 to 14 bits, Hartree-Fock word={molecule.hf_word}")
    print(f"Native trimmed Pauli terms={len(molecule.trimmed.terms)}, threshold=0.1")
    print(f"Actual objective queries={len(optimization.trace)}, angle count=30")
    print(
        f"Converged={optimization.converged}, query budget reached={optimization.query_budget_reached}"
    )
    print(f"Best actual trimmed energy={optimization.selected.energy}")
    print(
        f"Trimmed physical estimate={optimization.physical_trimmed_energy}, environment correction={molecule.environment_correction}"
    )
    total = optimization.physical_trimmed_energy + molecule.environment_correction
    print(
        f"WF-in-DFT total={total}, difference from DFT={total-molecule.dft_energy} Ha"
    )
    print(f"Full occupation-state errors={result.state_errors}")
    print(f"Independent full-Hamiltonian energy errors={result.energy_errors}")
    print(f"Actual untrimmed physical energy={optimization.physical_untrimmed_energy}")
    print(
        f"Source disabled diagonalization (historical only)={ARCHIVED_DISABLED_DIAGONALIZATION}"
    )


def write(result: workflow.Result, path: Path) -> None:
    molecule = result.molecule
    data = dict(
        basis="cc-pVDZ",
        ao_count=24,
        particles=[5, 5],
        fragment_electrons=10,
        environment_electrons=0,
        active_spatial_orbitals=8,
        width=14,
        parameters=30,
        query_budget=100,
        tolerance=1e-4,
        dft_energy=molecule.dft_energy,
        active_fci_energy=molecule.fci_energy,
        density_trace=molecule.density_trace,
        active_metric_error=molecule.active_metric_error,
        orbital_weights=molecule.orbital_weights,
        virtual_singular_values=molecule.virtual_singular_values,
        canonical_fermion_terms=len(molecule.fermion.terms),
        native_trimmed_terms=len(molecule.trimmed.terms),
        native_physical_trimmed_terms=len(molecule.physical_trimmed.terms),
        environment_correction=molecule.environment_correction,
        embedding_potential_norm=molecule.embedding_potential_norm,
        environment_projector_norm=molecule.environment_projector_norm,
        complete_state_errors=result.state_errors,
        independent_energy_errors=result.energy_errors,
        optimization=asdict(result.optimization),
        historical_disabled_diagonalization=ARCHIVED_DISABLED_DIAGONALIZATION,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("projection-embedding-results.json")
    )
    arguments = parser.parse_args()
    result = workflow.run()
    report(result)
    write(result, arguments.results)


if __name__ == "__main__":
    main()
