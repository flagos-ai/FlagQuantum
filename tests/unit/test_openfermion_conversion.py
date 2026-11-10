"""Real optional upstream operators checked against independent matrix action."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
import sympy
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.core.algebra.pauli import PauliOperator, PauliTerm
from flagquantum.ecosystem.openfermion import (
    from_openfermion_fermion,
    from_openfermion_qubit,
    to_openfermion_fermion,
    to_openfermion_qubit,
)
from flagquantum.observables.fermion import FermionOperator, FermionTerm
from tests.unit.fermion.oracles import fock_matrix

of = pytest.importorskip("openfermion")
pytestmark = pytest.mark.unit


def _ordered_matrix(
    terms: dict[tuple[tuple[int, int], ...], complex], width: int
) -> torch.Tensor:
    result = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for column in range(2**width):
        for word, coefficient in terms.items():
            bits = [(column >> (width - 1 - q)) & 1 for q in range(width)]
            amplitude = coefficient
            for mode, action in reversed(word):
                if bits[mode] == action:
                    amplitude = 0j
                    break
                amplitude *= (-1) ** sum(bits[:mode])
                bits[mode] = action
            row = sum(bit << (width - 1 - q) for q, bit in enumerate(bits))
            result[row, column] += amplitude
    return result


def test_all_short_ordered_words_and_complex_round_trips() -> None:
    for size in range(5):
        for word in itertools.product(((0, 0), (0, 1), (1, 0), (1, 1)), repeat=size):
            source = of.FermionOperator(word, 0.7 - 0.3j)
            before = dict(source.terms)
            native = from_openfermion_fermion(source)
            torch.testing.assert_close(
                fock_matrix(native, 2), _ordered_matrix(before, 2)
            )
            exported = to_openfermion_fermion(native)
            assert from_openfermion_fermion(exported) == native
            # Compare actual upstream actions, not its different normal-order keys.
            torch.testing.assert_close(
                torch.tensor(
                    of.get_sparse_operator(exported, n_qubits=2).toarray(),
                    dtype=torch.complex128,
                ),
                torch.tensor(
                    of.get_sparse_operator(source, n_qubits=2).toarray(),
                    dtype=torch.complex128,
                ),
            )
            assert source.terms == before


def test_pauli_complex_sum_against_upstream_matrix_and_independent_factors() -> None:
    source = of.QubitOperator("X0 Y2", 0.5j) + of.QubitOperator("Z1", -2)
    source += of.QubitOperator((), 0.3 - 0.2j)
    native = from_openfermion_qubit(source)
    matrix = pauli_matrix(native, n_qubits=3)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    eye = torch.eye(2, dtype=torch.complex128)
    expected = 0.5j * torch.kron(torch.kron(x, eye), y)
    expected -= 2 * torch.kron(torch.kron(eye, z), eye)
    expected += (0.3 - 0.2j) * torch.eye(8, dtype=torch.complex128)
    torch.testing.assert_close(matrix, expected)
    torch.testing.assert_close(
        matrix, torch.tensor(of.get_sparse_operator(source, n_qubits=3).toarray())
    )
    exported = to_openfermion_qubit(native)
    assert from_openfermion_qubit(exported) == native
    exported.terms.clear()
    assert native.terms


def test_tiny_coefficients_identity_zero_and_ownership() -> None:
    fermion = FermionOperator((FermionTerm(1e-30, (0,), ()), FermionTerm(2j)))
    pauli = PauliOperator((PauliTerm(1e-30, ((0, "x"),)), PauliTerm(2j)))
    assert from_openfermion_fermion(to_openfermion_fermion(fermion)) == fermion
    assert from_openfermion_qubit(to_openfermion_qubit(pauli)) == pauli
    assert from_openfermion_fermion(of.FermionOperator()) == FermionOperator()
    assert from_openfermion_qubit(of.QubitOperator()) == PauliOperator()
    assert to_openfermion_fermion(FermionOperator()).terms == {}
    assert to_openfermion_qubit(PauliOperator()).terms == {}
    source = of.FermionOperator("0 0^", 2)
    native = from_openfermion_fermion(source)
    source.terms.clear()
    assert fock_matrix(native, 1)[0, 0].item() == 2


def test_converted_operator_uses_unchanged_measurement_consumer() -> None:
    source = of.FermionOperator("0^ 0", 1.5) + of.FermionOperator("1^ 1", -0.5)
    source += of.FermionOperator((), 0.25)
    native = from_openfermion_fermion(source)
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(native, n_modes=3, encoding=encoding)
        circuit = fq.Circuit(3, dtype=torch.complex128)
        for q, bit in enumerate(encoded.reference_bits((1, 1, 0))):
            if bit:
                circuit.x(q)
        value = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        assert float(
            torch.as_tensor(value.expectation()).reshape(-1)[0]
        ) == pytest.approx(1.25)


@pytest.mark.parametrize("convert", [from_openfermion_fermion, from_openfermion_qubit])
@pytest.mark.parametrize("source", [None, {}, 1, FermionOperator(), PauliOperator()])
def test_wrong_external_type(
    convert: Callable[[object], object], source: object
) -> None:
    with pytest.raises(TypeError):
        convert(source)


@pytest.mark.parametrize("convert", [to_openfermion_fermion, to_openfermion_qubit])
def test_wrong_native_type(convert: Callable[[object], object]) -> None:
    with pytest.raises(TypeError):
        convert(of.FermionOperator())


@pytest.mark.parametrize("kind", ["fermion", "qubit"])
@pytest.mark.parametrize(
    "coefficient",
    [
        True,
        float("nan"),
        float("inf"),
        1j * float("inf"),
        "x",
        sympy.Symbol("x"),
        torch.tensor(1.0),
        torch.tensor(1.0, requires_grad=True),
    ],
)
def test_invalid_external_coefficient(kind: str, coefficient: object) -> None:
    source = of.FermionOperator() if kind == "fermion" else of.QubitOperator()
    source.terms = {(): coefficient}
    convert = from_openfermion_fermion if kind == "fermion" else from_openfermion_qubit
    with pytest.raises((TypeError, ValueError)):
        convert(source)


@pytest.mark.parametrize(
    "kind,word",
    [
        ("fermion", ((0, 2),)),
        ("fermion", ((0, True),)),
        ("fermion", ((-1, 0),)),
        ("fermion", ((True, 0),)),
        ("fermion", ((0.0, 1),)),
        ("fermion", ((0,),)),
        ("qubit", ((0, "A"),)),
        ("qubit", ((0, "x"),)),
        ("qubit", ((0, "X"), (0, "Y"))),
        ("qubit", ((-1, "Z"),)),
        ("qubit", ((True, "Z"),)),
        ("qubit", "X0"),
    ],
)
def test_corrupted_term_dictionary(kind: str, word: object) -> None:
    source = of.FermionOperator() if kind == "fermion" else of.QubitOperator()
    source.terms = {word: 1}
    convert = from_openfermion_fermion if kind == "fermion" else from_openfermion_qubit
    with pytest.raises((TypeError, ValueError)):
        convert(source)


def test_corrupted_mapping_and_wrong_upstream_operator_kind() -> None:
    source = of.FermionOperator()
    source.terms = []
    with pytest.raises(TypeError, match="mapping"):
        from_openfermion_fermion(source)
    with pytest.raises(TypeError):
        from_openfermion_fermion(of.QubitOperator())
    with pytest.raises(TypeError):
        from_openfermion_qubit(of.FermionOperator())
