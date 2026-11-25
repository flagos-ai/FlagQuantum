"""Sector reduction compared to direct affine embeddings."""

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.symmetry import reduce_z_parities
from flagquantum.observables import annihilate, create, number

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sector", [-1, 1])
def test_sector_matrix_and_state_embedding(sector: int) -> None:
    hop = create(0) * annihilate(2)
    encoded = encode_fermion(hop + hop.dagger() + 0.4 * number(1), n_modes=3)
    reduced = reduce_z_parities(encoded, generators=[(0, 1, 2)], eigenvalues=[sector])
    embedding = torch.stack(
        [
            reduced.mapping.lift_state(column)
            for column in torch.eye(4, dtype=torch.complex128)
        ],
        dim=1,
    )
    full = pauli_matrix(encoded.operator, n_qubits=3)
    actual = pauli_matrix(reduced.operator, n_qubits=2)
    torch.testing.assert_close(full @ embedding, embedding @ actual)
    for column in embedding.T:
        index = int(column.abs().argmax())
        assert (-1) ** index.bit_count() == sector


def test_multiple_constraints_and_zero_qubit_sector() -> None:
    encoded = encode_fermion(number(0) + 2 * number(1), n_modes=2)
    reduced = reduce_z_parities(encoded, generators=[(0, 1), (1,)], eigenvalues=[-1, 1])
    assert reduced.n_qubits == 0
    assert reduced.mapping.full_bits(()) == (1, 0)
    assert pauli_matrix(reduced.operator, n_qubits=0).item() == pytest.approx(1)


def test_rejects_nonpreserving_and_dependent_inputs() -> None:
    encoded = encode_fermion(create(0), n_modes=2)
    with pytest.raises(ValueError, match="preserve"):
        reduce_z_parities(encoded, generators=[(0,)], eigenvalues=[1])
    with pytest.raises(ValueError, match="independent"):
        reduce_z_parities(encoded, generators=[(0,), (0,)], eigenvalues=[1, 1])
