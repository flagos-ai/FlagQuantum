"""Fresh original cc-pVDZ projection embedding with source-defined virtuals."""

from dataclasses import dataclass

import numpy as np
import torch

from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.stabilizer import (
    StabilizerReduction,
    reduce_stabilizers,
)
from flagquantum.algorithms.molecular import second_quantized_hamiltonian
from flagquantum.core.algebra.pauli import PauliOperator, pauli
from flagquantum.observables.fermion import FermionOperator

WIDTH, OCCUPIED, VIRTUALS = 16, 5, 3
THRESHOLD, MU, WEIGHT_CUTOFF = 0.1, 1e6, 0.3
GEOMETRY = "O 0 0 0; H 0 -.757 .587; H 0 .757 .587"


@dataclass(frozen=True)
class Model:
    one_body: torch.Tensor
    two_body: torch.Tensor
    nuclear_energy: float
    dft_energy: float
    fci_energy: float
    density_trace: float
    orbital_weights: tuple[float, ...]
    virtual_singular_values: tuple[float, ...]
    active_metric_error: float
    fermion: FermionOperator
    physical_fermion: FermionOperator
    full_qubit: PauliOperator
    reduction: StabilizerReduction
    trimmed: PauliOperator
    physical_trimmed: PauliOperator
    hf_word: int
    environment_correction: float
    embedding_potential_norm: float
    environment_projector_norm: float


def trim(operator: PauliOperator, threshold: float) -> PauliOperator:
    return PauliOperator(
        tuple(term for term in operator.terms if abs(term.coefficient) > threshold)
    )


def regenerate() -> Model:
    """Keep all 24 AOs, ten electrons and eight source active spatial orbitals."""
    from pyscf import ao2mo, dft, fci, gto

    molecule = gto.M(
        atom=GEOMETRY, basis="cc-pVDZ", charge=0, spin=0, unit="Angstrom", verbose=0
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "B3LYP"
    mean_field.conv_tol = 1e-9
    mean_field.max_memory = 512
    mean_field.chkfile = None
    dft_energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError("The original water DFT did not converge")
    overlap = mean_field.get_ovlp()
    occupied = mean_field.mo_coeff[:, mean_field.mo_occ > 0]
    first, last = molecule.aoslice_by_atom()[0, 2:]
    weights = np.sum(occupied[first:last] * (overlap @ occupied)[first:last], axis=0)
    if occupied.shape[1] != OCCUPIED or not np.all(weights > WEIGHT_CUTOFF):
        raise ValueError("The original oxygen fragment requires all five occupied MOs")
    # With all electrons in A, D_B, v_emb, mu P_B and E_environment vanish exactly.
    environment_density = np.zeros_like(overlap)
    projector = overlap @ environment_density @ overlap
    embedding = np.zeros_like(overlap)
    virtual = mean_field.mo_coeff[:, mean_field.mo_occ == 0]
    coupling = virtual.T @ overlap @ mean_field.get_fock() @ occupied
    left, singular, _ = np.linalg.svd(coupling, full_matrices=False)
    active = np.column_stack((occupied, virtual @ left[:, :VIRTUALS]))
    metric_error = float(np.max(np.abs(active.T @ overlap @ active - np.eye(8))))
    if metric_error > 1e-12:
        raise ValueError("The original active orbital basis is not S-orthonormal")
    core = mean_field.get_hcore()
    physical_one = active.T @ core @ active
    embedded_one = active.T @ (core + embedding + MU * projector) @ active
    two = ao2mo.kernel(molecule, active, compact=False).reshape(8, 8, 8, 8)
    nuclear = float(molecule.energy_nuc())
    solver = fci.direct_spin1.FCI(molecule)
    solver.conv_tol = 1e-10
    solver.max_memory = 512
    energy, _ = solver.kernel(embedded_one, two, 8, (5, 5), ecore=nuclear)
    if not solver.converged:
        raise RuntimeError("The original active-space FCI did not converge")
    fermion = second_quantized_hamiltonian(
        torch.as_tensor(embedded_one, dtype=torch.float64),
        torch.as_tensor(two, dtype=torch.float64),
        nuclear,
    )
    if not np.array_equal(embedded_one, physical_one):
        raise ValueError(
            "The original zero-environment case must have identical Hamiltonians"
        )
    # These roles are exactly equal for the source's empty environment; encode once.
    physical = fermion
    hamiltonian = trim(encode_fermion(fermion, n_modes=WIDTH).operator, 1e-12)
    generators = tuple(
        pauli(factors=tuple((wire, "z") for wire in range(spin, WIDTH, 2)))
        for spin in range(2)
    )
    reduction = reduce_stabilizers(
        hamiltonian, n_qubits=WIDTH, generators=generators, eigenvalues=(-1, -1)
    )
    word = ((1 << 10) - 1) << 6
    for gate in reduction.clifford:
        if gate.name != "cx":
            raise ValueError(
                "Spin parity requires a computational-basis Clifford frame"
            )
        control, target = gate.wires
        if word & (1 << (WIDTH - control - 1)):
            word ^= 1 << (WIDTH - target - 1)
    reduced_word = 0
    for wire in reduction.mapping.free_qubits:
        reduced_word = 2 * reduced_word + ((word >> (WIDTH - wire - 1)) & 1)
    physical_qubit = hamiltonian
    return Model(
        torch.as_tensor(embedded_one, dtype=torch.float64),
        torch.as_tensor(two, dtype=torch.float64),
        nuclear,
        dft_energy,
        float(energy),
        float(np.trace(mean_field.make_rdm1() @ overlap)),
        tuple(float(value) for value in weights),
        tuple(float(value) for value in singular),
        metric_error,
        fermion,
        physical,
        hamiltonian,
        reduction,
        trim(reduction.operator, THRESHOLD),
        trim(reduction.reduce_operator(physical_qubit), THRESHOLD),
        reduced_word,
        0.0,
        float(np.linalg.norm(embedding)),
        float(np.linalg.norm(projector)),
    )
