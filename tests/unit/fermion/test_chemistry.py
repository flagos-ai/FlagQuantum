"""Native integral conventions, reference occupations and active projections."""

import pytest
import torch

from flagquantum.algorithms.chemistry_domain.active_space import select_active_space
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.molecular import create_molecular_hamiltonian

from .oracles import fock_matrix

pytestmark = pytest.mark.unit


def test_existing_hydrogen_problem_and_total_energy_offsets() -> None:
    native = create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
    problem = from_native_molecule(native)
    operator = fermion_hamiltonian(problem)
    full = fock_matrix(operator, problem.n_modes)
    assert float(torch.linalg.eigvalsh(full)[0]) == pytest.approx(
        native.fci_energy, abs=1e-8
    )
    for encoding in ENCODINGS:
        encoded = encode_fermion(operator, n_modes=problem.n_modes, encoding=encoding)
        bits = encoded.reference_bits(problem.reference_occupation)
        index = sum(bit << (len(bits) - i - 1) for i, bit in enumerate(bits))
        matrix = pauli_matrix(encoded.operator, n_qubits=problem.n_modes)
        assert float(matrix[index, index].real) == pytest.approx(
            native.hf_energy, abs=1e-9
        )


def test_active_space_is_fixed_occupancy_projection_with_reordering() -> None:
    generator = torch.Generator().manual_seed(17)
    n = 3
    factor = torch.randn(n, n, generator=generator, dtype=torch.float64)
    factor = (factor + factor.T) / 2
    two = torch.einsum("pq,rs->pqrs", factor, factor)
    problem = MolecularProblem(
        torch.diag(torch.tensor([-2.0, -1.0, 0.7], dtype=torch.float64)),
        two,
        4,
        0.3,
        0.2,
    )
    selected = select_active_space(
        problem, active_orbitals=(2, 0), frozen_occupied=(1,)
    )
    active = selected.problem
    assert active.reference_occupation == (0, 0, 1, 1)
    assert active.orbital_labels == (2, 0)
    full = fock_matrix(fermion_hamiltonian(problem), 6)
    indices = [i for i in range(64) if i & 12 == 12]
    restricted = full[indices][:, indices]
    reduced = fock_matrix(fermion_hamiltonian(active), 4)
    torch.testing.assert_close(
        torch.linalg.eigvalsh(restricted), torch.linalg.eigvalsh(reduced)
    )
    full_reference = int("111100", 2)
    assert float(full[full_reference, full_reference].real) == pytest.approx(
        float(reduced[3, 3].real)
    )
    assert problem.n_electrons == 4
    assert problem.core_energy == 0.2


def test_static_problem_validation_and_snapshot_ownership() -> None:
    one = torch.eye(2, dtype=torch.float64)
    two = torch.zeros(2, 2, 2, 2, dtype=torch.float64)
    problem = MolecularProblem(one, two, 2)
    one.zero_()
    assert problem.one_body.trace() == 2
    with pytest.raises(ValueError, match="autograd"):
        MolecularProblem(one.requires_grad_(), two, 2)
    with pytest.raises(ValueError, match="occupied"):
        select_active_space(problem, active_orbitals=(1,))
    with pytest.raises(ValueError, match="distinct"):
        select_active_space(problem, active_orbitals=(0, 0))
