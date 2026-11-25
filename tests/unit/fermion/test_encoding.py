"""Algebra, basis equivalence and the same native energy consumer."""

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.core.algebra.pauli import PauliOperator, PauliTerm, pauli
from flagquantum.observables import annihilate, create, number
from flagquantum.observables.fermion import FermionOperator

from .oracles import fock_matrix

pytestmark = pytest.mark.unit


def test_pauli_products_and_adjoint() -> None:
    x = pauli(factors=((0, "x"),))
    y = pauli(factors=((0, "y"),))
    z = pauli(factors=((0, "z"),))
    assert x * y == 1j * z
    assert y * x == -1j * z
    assert (x + 2j * y).dagger() == x - 2j * y
    assert x * x == pauli()
    assert x - x == PauliOperator()
    assert PauliOperator((PauliTerm(1j), PauliTerm(-1j), PauliTerm(2))).is_hermitian()
    with pytest.raises(ValueError, match="non-Hermitian"):
        to_observable(1j * x)
    assert to_observable(PauliOperator()).terms[0].coefficient == 0


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6])
def test_encoding_matches_independent_fock_action(
    encoding: Encoding, width: int
) -> None:
    operator = (1.2 + 0.7j) * create(width - 1) + 0.3 * number(0)
    if width > 1:
        operator = operator + (0.2 - 0.4j) * create(0) * annihilate(width - 1)
    encoded = encode_fermion(operator, n_modes=width, encoding=encoding)
    permutation = []
    for index in range(2**width):
        bits = tuple((index >> (width - i - 1)) & 1 for i in range(width))
        mapped = encoded.reference_bits(bits)
        permutation.append(sum(bit << (width - i - 1) for i, bit in enumerate(mapped)))
    image = pauli_matrix(encoded.operator, n_qubits=width)
    torch.testing.assert_close(
        image[permutation][:, permutation], fock_matrix(operator, width)
    )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_one_unchanged_native_measurement_consumer(encoding: Encoding) -> None:
    hamiltonian = 1.3 * number(0) - 0.4 * number(1)
    encoded = encode_fermion(hamiltonian, n_modes=3, encoding=encoding)
    circuit = fq.Circuit(encoded.n_qubits)
    for wire, bit in enumerate(encoded.reference_bits((1, 0, 0))):
        if bit:
            circuit.x(wire)
    result = fq.run(circuit, outputs=[fq.expectation(to_observable(encoded.operator))])
    assert float(torch.as_tensor(result.expectation()).reshape(-1)[0]) == pytest.approx(
        1.3
    )


def test_width_identity_and_rejections() -> None:
    encoded = encode_fermion(FermionOperator(), n_modes=4)
    assert encoded.n_qubits == 4
    assert not encoded.operator.terms
    with pytest.raises(ValueError, match="outside"):
        encode_fermion(number(3), n_modes=3)
    with pytest.raises(TypeError):
        encode_fermion(number(0), n_modes=True)
    with pytest.raises(ValueError, match="occupation"):
        encoded.reference_bits((1, 0))
    with pytest.raises(ValueError, match="binary"):
        encoded.reference_bits((0, 2, 0, 0))
    with pytest.raises(ValueError, match="max_qubits"):
        pauli_matrix(pauli(), n_qubits=20)
    with pytest.raises(ValueError, match="repeat"):
        pauli(factors=((0, "x"), (0, "y")))
    with pytest.raises(ValueError, match="finite"):
        pauli(float("nan"))
