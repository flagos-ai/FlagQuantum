"""Build each hydrogen molecule and its paired-electron symmetry sector."""

from dataclasses import dataclass

import torch

from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.molecular import create_molecular_hamiltonian

DTYPE = torch.float64
SECTOR = (0b0011, 0b1100)


@dataclass(frozen=True)
class Model:
    hamiltonian: torch.Tensor
    rhf: float
    fci: float
    coupling_to_complement: float


def distances() -> tuple[float, ...]:
    short = torch.linspace(0.20, 1.0, 5, dtype=DTYPE)
    long = torch.linspace(1.4, 3.5, 7, dtype=DTYPE)
    return tuple(map(float, torch.cat((short, long))))


def build_model(distance: float) -> Model:
    geometry = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distance)))
    molecule = create_molecular_hamiltonian(geometry, basis="sto-3g")
    problem = from_native_molecule(molecule)
    encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=4)
    full = pauli_matrix(encoded.operator, n_qubits=4)
    paired = list(SECTOR)
    complement = [word for word in range(16) if word not in SECTOR]
    coupling = float(full[paired][:, complement].abs().max())
    if coupling >= 1e-10:
        raise ValueError("The paired determinants must form an invariant sector")
    # The retained native ansatz starts at |1>. Its corresponding determinant
    # occupies both spins of the lower-energy spatial orbital.
    tapered = full[paired][:, paired]
    if molecule.fci_energy is None:
        raise ValueError("An exact full-space reference is required for this example")
    return Model(tapered, molecule.hf_energy, molecule.fci_energy, coupling)
