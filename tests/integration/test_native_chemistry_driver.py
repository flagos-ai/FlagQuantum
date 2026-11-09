"""The direct chemistry boundary produces native, interchangeable problem data."""

import importlib
import importlib.util

import pytest
import torch

from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.ecosystem.chemistry.pyscf import build_molecular_problem

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("pyscf") is None,
        reason="install flagquantum[chemistry-pyscf] for direct RHF tests",
    ),
]


@pytest.fixture(autouse=True)
def require_pyscf() -> None:
    # An installed but broken driver must fail instead of hiding behind a skip.
    importlib.import_module("pyscf")


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_direct_pyscf_reference_energy(basis: str) -> None:
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    problem = build_molecular_problem(geometry, basis=basis)
    encoded = encode_fermion(fermion_hamiltonian(problem), n_modes=problem.n_modes)
    matrix = pauli_matrix(encoded.operator, n_qubits=problem.n_modes)
    bits = problem.reference_occupation
    index = sum(bit << (len(bits) - i - 1) for i, bit in enumerate(bits))
    assert float(matrix[index, index].real) == pytest.approx(
        problem.reference_energy, abs=1e-8
    )
    if basis == "sto-3g":
        native = from_native_molecule(create_molecular_hamiltonian(geometry))
        reference = pauli_matrix(
            encode_fermion(fermion_hamiltonian(native), n_modes=4).operator, n_qubits=4
        )
        torch.testing.assert_close(
            torch.linalg.eigvalsh(matrix),
            torch.linalg.eigvalsh(reference),
            atol=1e-7,
            rtol=1e-7,
        )


def test_lithium_hydride_hf_expectation() -> None:
    problem = build_molecular_problem([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))])
    encoded = encode_fermion(
        fermion_hamiltonian(problem), n_modes=problem.n_modes, encoding="bravyi_kitaev"
    )
    bits = encoded.reference_bits(problem.reference_occupation)
    # Basis expectation needs no dense 4096-square Hamiltonian.
    energy = 0j
    for term in encoded.operator.terms:
        if all(axis == "z" for _, axis in term.factors):
            energy += term.coefficient * (-1) ** sum(bits[q] for q, _ in term.factors)
    assert energy.real == pytest.approx(problem.reference_energy, abs=1e-8)
