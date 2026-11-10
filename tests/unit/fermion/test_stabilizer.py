"""Stabilizer sectors against independent projectors and native Clifford matrices."""

from __future__ import annotations

import itertools

import pytest
import torch

from flagquantum.algorithms.fermion.clifford import conjugate_pauli
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.lowering import (
    lift_stabilizer_state,
    stabilizer_clifford,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.stabilizer import reduce_stabilizers
from flagquantum.algorithms.fermion.symmetry import reduce_z_parities
from flagquantum.circuit import Circuit
from flagquantum.core.algebra.pauli import PauliOperator, pauli
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.observables import annihilate, create, number
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


def word(axes: str, sign: int = 1) -> PauliOperator:
    return pauli(sign, tuple((q, axis) for q, axis in enumerate(axes) if axis != "i"))


def projector(
    generators: list[PauliOperator], sectors: tuple[int, ...], n: int
) -> torch.Tensor:
    identity = torch.eye(2**n, dtype=torch.complex128)
    result = identity.clone()
    for generator, sector in zip(generators, sectors, strict=True):
        result = result @ (identity + sector * pauli_matrix(generator, n_qubits=n)) / 2
    return result


@pytest.mark.parametrize(
    "gate",
    [
        Instruction("h", (0,)),
        Instruction("h", (1,)),
        Instruction("sdg", (0,)),
        Instruction("sdg", (1,)),
        Instruction("cx", (0, 1)),
        Instruction("cx", (1, 0)),
    ],
)
def test_clifford_images_include_all_y_signs(gate: Instruction) -> None:
    unitary = get_unitary(
        Circuit.from_ir(CircuitIR(n_wires=2, dtype="complex128", instructions=(gate,)))
    )
    for axes in itertools.product("ixyz", repeat=2):
        operator = word("".join(axes))
        actual = pauli_matrix(conjugate_pauli(operator, gate), n_qubits=2)
        expected = unitary @ pauli_matrix(operator, n_qubits=2) @ unitary.mH
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "axes",
    [("xx", "zz"), ("-xx", "yy"), ("yii", "ixz"), ("xyiz", "yxzi", "zzii"), ("y",), ()],
)
def test_all_joint_sectors_against_dense_projectors(axes: tuple[str, ...]) -> None:
    n = len(axes[0].lstrip("-")) if axes else 3
    generators = [word(a.lstrip("-"), -1 if a.startswith("-") else 1) for a in axes]
    matrices = [pauli_matrix(g, n_qubits=n) for g in generators]
    # Build a Hamiltonian from the entire commutant, selected with dense matrix
    # commutators rather than the production binary symplectic predicate.
    operator = PauliOperator()
    for index, labels in enumerate(itertools.product("ixyz", repeat=n)):
        candidate = word("".join(labels))
        matrix = pauli_matrix(candidate, n_qubits=n)
        if all(torch.allclose(g @ matrix, matrix @ g) for g in matrices):
            operator = operator + ((index % 13 - 6) / 17) * candidate
    full = pauli_matrix(operator, n_qubits=n)
    for sectors in itertools.product((-1, 1), repeat=len(generators)):
        reduced = reduce_stabilizers(
            operator, n_qubits=n, generators=generators, eigenvalues=sectors
        )
        embedding = torch.stack(
            [
                lift_stabilizer_state(reduced, v)
                for v in torch.eye(2**reduced.n_qubits, dtype=torch.complex128)
            ],
            dim=1,
        )
        expected_projector = projector(generators, sectors, n)
        torch.testing.assert_close(
            embedding @ embedding.mH, expected_projector, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            embedding.mH @ embedding,
            torch.eye(2**reduced.n_qubits, dtype=torch.complex128),
            atol=1e-12,
            rtol=1e-12,
        )
        actual = pauli_matrix(reduced.operator, n_qubits=reduced.n_qubits)
        torch.testing.assert_close(
            full @ embedding, embedding @ actual, atol=1e-12, rtol=1e-12
        )
        _, vectors = torch.linalg.eigh(expected_projector)
        basis = vectors[:, -(2**reduced.n_qubits) :]
        torch.testing.assert_close(
            torch.linalg.eigvalsh(actual),
            torch.linalg.eigvalsh(basis.mH @ full @ basis),
            atol=1e-12,
            rtol=1e-12,
        )
        for g, sector in zip(matrices, sectors, strict=True):
            torch.testing.assert_close(
                g @ embedding, sector * embedding, atol=1e-12, rtol=1e-12
            )


def test_shared_frame_observables_and_unnormalized_complex_amplitudes() -> None:
    generator = word("yx")
    observable = word("xy") + 0.3 * word("zz")
    result = reduce_stabilizers(
        generator, n_qubits=2, generators=[generator], eigenvalues=[-1]
    )
    reduced_observable = result.reduce_operator(observable)
    vector = torch.tensor([0.7 + 2j, -0.4j], dtype=torch.complex128)
    full_vector = lift_stabilizer_state(result, vector)
    torch.testing.assert_close(
        torch.vdot(full_vector, full_vector), torch.vdot(vector, vector)
    )
    torch.testing.assert_close(
        torch.vdot(full_vector, pauli_matrix(observable, n_qubits=2) @ full_vector),
        torch.vdot(vector, pauli_matrix(reduced_observable, n_qubits=1) @ vector),
    )
    torch.testing.assert_close(
        lift_stabilizer_state(result, 1j * vector), 1j * full_vector
    )
    with pytest.raises(ValueError, match="preserve"):
        result.reduce_operator(word("zi"))
    with pytest.raises(ValueError, match="outside"):
        result.reduce_operator(word("iix"))


@pytest.mark.parametrize("sector", [-1, 1])
def test_diagonal_replacement_retains_physical_sector(sector: int) -> None:
    hopping = create(0) * annihilate(2)
    encoded = encode_fermion(hopping + hopping.dagger() + 0.4 * number(1), n_modes=3)
    old = reduce_z_parities(encoded, generators=[(0, 1, 2)], eigenvalues=[sector])
    new = reduce_stabilizers(
        encoded.operator, n_qubits=3, generators=[word("zzz")], eigenvalues=[sector]
    )
    torch.testing.assert_close(
        torch.linalg.eigvalsh(pauli_matrix(old.operator, n_qubits=2)),
        torch.linalg.eigvalsh(pauli_matrix(new.operator, n_qubits=2)),
    )
    left = torch.stack(
        [old.mapping.lift_state(v) for v in torch.eye(4, dtype=torch.complex128)], dim=1
    )
    right = torch.stack(
        [lift_stabilizer_state(new, v) for v in torch.eye(4, dtype=torch.complex128)],
        dim=1,
    )
    torch.testing.assert_close(left @ left.mH, right @ right.mH)


def test_empty_and_zero_qubit_registers_and_complex_operator() -> None:
    operator = (1 + 2j) * word("yx")
    result = reduce_stabilizers(operator, n_qubits=2, generators=[], eigenvalues=[])
    assert result.operator == operator
    vector = torch.tensor([0, 1j, -2, 0.4], dtype=torch.complex128)
    torch.testing.assert_close(lift_stabilizer_state(result, vector), vector)
    zero = reduce_stabilizers(pauli(2j), n_qubits=0, generators=[], eigenvalues=[])
    assert zero.n_qubits == 0
    torch.testing.assert_close(
        lift_stabilizer_state(zero, torch.tensor([3j])),
        torch.tensor([3j], dtype=torch.complex128),
    )
    with pytest.raises(ValueError, match="zero-qubit"):
        stabilizer_clifford(zero)


@pytest.mark.parametrize(
    "generators,values,error",
    [
        ([word("xx"), word("xx")], [1, 1], "independent"),
        ([word("xx"), -word("xx")], [1, -1], "independent"),
        ([word("xx"), word("zi")], [1, 1], "commute"),
        ([word("xx"), word("yy"), word("zz")], [1, 1, 1], "independent"),
        ([pauli()], [1], "nonidentity"),
        ([PauliOperator()], [1], "one signed"),
        ([2 * word("x")], [1], "coefficient"),
        ([1j * word("x")], [1], "coefficient"),
        ([word("x") + word("z")], [1], "one signed"),
        ([word("iix")], [1], "outside"),
        ([word("x")], [], "one eigenvalue"),
        ([word("x")], [0], "eigenvalues"),
    ],
)
def test_invalid_stabilizers(
    generators: list[PauliOperator], values: list[int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        reduce_stabilizers(
            pauli(), n_qubits=2, generators=generators, eigenvalues=values
        )


def test_nonpreserving_operator_and_strict_input_types() -> None:
    with pytest.raises(ValueError, match="preserve"):
        reduce_stabilizers(
            word("x"), n_qubits=1, generators=[word("z")], eigenvalues=[1]
        )
    for invalid in (True, 1.0, "1", None):
        with pytest.raises(TypeError):
            reduce_stabilizers(
                pauli(), n_qubits=1, generators=[word("x")], eigenvalues=[invalid]
            )
    with pytest.raises(TypeError, match="native"):
        reduce_stabilizers("x", n_qubits=1, generators=[], eigenvalues=[])
    with pytest.raises(TypeError, match="native"):
        reduce_stabilizers(pauli(), n_qubits=1, generators=["x"], eigenvalues=[1])
    with pytest.raises(ValueError, match="outside"):
        reduce_stabilizers(word("ix"), n_qubits=1, generators=[], eigenvalues=[])
    with pytest.raises(ValueError, match="n_qubits"):
        reduce_stabilizers(pauli(), n_qubits=-1, generators=[], eigenvalues=[])
    with pytest.raises(TypeError):
        reduce_stabilizers(pauli(), n_qubits=True, generators=[], eigenvalues=[])
    with pytest.raises(ValueError, match="supports"):
        conjugate_pauli(pauli(), Instruction("x", (0,)))


def test_lift_validation_and_budget_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = reduce_stabilizers(
        word("x"), n_qubits=1, generators=[word("x")], eigenvalues=[1]
    )
    for state in (
        torch.ones(2),
        torch.ones(1, 1),
        torch.tensor([float("nan")]),
        torch.ones(1, requires_grad=True),
    ):
        with pytest.raises(ValueError):
            lift_stabilizer_state(result, state)
    with pytest.raises(TypeError, match="numeric"):
        lift_stabilizer_state(result, torch.tensor([True]))
    with pytest.raises(TypeError, match="torch.Tensor"):
        lift_stabilizer_state(result, [1])
    with pytest.raises(TypeError, match="StabilizerReduction"):
        lift_stabilizer_state(None, torch.ones(1))
    with pytest.raises(TypeError, match="StabilizerReduction"):
        stabilizer_clifford(None)
    with pytest.raises(TypeError, match="inverse"):
        stabilizer_clifford(result, inverse=1)
    large = reduce_stabilizers(
        pauli(), n_qubits=21, generators=[word("x")], eigenvalues=[1]
    )

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("allocation attempted before width guard")

    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(ValueError, match="20 qubits"):
        lift_stabilizer_state(large, torch.ones(1))


def test_clifford_and_adjoint_match_the_symbolic_frame() -> None:
    result = reduce_stabilizers(
        word("xyz"), n_qubits=3, generators=[word("xyz")], eigenvalues=[-1]
    )
    c = get_unitary(stabilizer_clifford(result))
    inverse = get_unitary(stabilizer_clifford(result, inverse=True))
    torch.testing.assert_close(inverse, c.mH)
    diagonal = -pauli_matrix(
        pauli(factors=((result.mapping.pivots[0], "z"),)), n_qubits=3
    )
    torch.testing.assert_close(
        c @ pauli_matrix(-word("xyz"), n_qubits=3) @ c.mH, diagonal
    )


@pytest.mark.parametrize("seed", range(5))
def test_random_native_clifford_frames_with_signed_generators(seed: int) -> None:
    n = 3
    generator = torch.Generator().manual_seed(seed + 419)
    circuit = Circuit(n, dtype=torch.complex128)
    for _ in range(12):
        q = int(torch.randint(n, (), generator=generator))
        choice = int(torch.randint(3, (), generator=generator))
        if choice == 0:
            circuit.h(q)
        elif choice == 1:
            circuit.s(q)
        else:
            circuit.cnot(q, (q + 1) % n)
    c = get_unitary(circuit)
    # Recover each rotated word from a complete independent dense Pauli basis;
    # this oracle never calls the production symbolic conjugation routine.
    basis = [
        (word("".join(axes)), pauli_matrix(word("".join(axes)), n_qubits=n))
        for axes in itertools.product("ixyz", repeat=n)
    ]

    def recover(matrix: torch.Tensor) -> PauliOperator:
        result = PauliOperator()
        for candidate, dense in basis:
            coefficient = complex(torch.trace(dense @ matrix) / 2**n)
            if abs(coefficient) > 1e-10:
                result = result + round(coefficient.real) * candidate
        return result

    generators = [
        recover(c @ pauli_matrix(word("zii"), n_qubits=n) @ c.mH),
        -recover(c @ pauli_matrix(word("izi"), n_qubits=n) @ c.mH),
    ]
    initial = word("zii") + 2 * word("izx") + 0.3 * word("iiy")
    # Coefficients are not rounded for the Hamiltonian.
    dense = c @ pauli_matrix(initial, n_qubits=n) @ c.mH
    operator = PauliOperator()
    for candidate, matrix in basis:
        coefficient = complex(torch.trace(matrix @ dense) / 2**n)
        if abs(coefficient) > 1e-10:
            operator = operator + coefficient.real * candidate
    for sectors in itertools.product((-1, 1), repeat=2):
        result = reduce_stabilizers(
            operator, n_qubits=n, generators=generators, eigenvalues=sectors
        )
        embedding = torch.stack(
            [
                lift_stabilizer_state(result, v)
                for v in torch.eye(2, dtype=torch.complex128)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            embedding @ embedding.mH,
            projector(generators, sectors, n),
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            dense @ embedding,
            embedding @ pauli_matrix(result.operator, n_qubits=1),
            atol=1e-12,
            rtol=1e-12,
        )
        reordered = reduce_stabilizers(
            operator,
            n_qubits=n,
            generators=list(reversed(generators)),
            eigenvalues=tuple(reversed(sectors)),
        )
        torch.testing.assert_close(
            torch.linalg.eigvalsh(pauli_matrix(result.operator, n_qubits=1)),
            torch.linalg.eigvalsh(pauli_matrix(reordered.operator, n_qubits=1)),
            atol=1e-12,
            rtol=1e-12,
        )


def test_symbolic_conjugation_rejects_matrix_overrides() -> None:
    gate = Instruction("h", (0,), matrix=torch.eye(2, dtype=torch.complex128))
    with pytest.raises(ValueError, match="fixed"):
        conjugate_pauli(word("x"), gate)


@pytest.mark.parametrize("encoding", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_supplied_symmetry_uses_the_operators_encoded_basis(encoding: str) -> None:
    hopping = create(0) * annihilate(2)
    full = encode_fermion(
        hopping + hopping.dagger() + 0.4 * number(1), n_modes=3, encoding=encoding
    )
    identity = annihilate(0) * create(0) + number(0)
    parity = (
        (identity - 2 * number(0))
        * (identity - 2 * number(1))
        * (identity - 2 * number(2))
    )
    symmetry = encode_fermion(parity, n_modes=3, encoding=encoding).operator
    result = reduce_stabilizers(
        full.operator, n_qubits=3, generators=[symmetry], eigenvalues=[-1]
    )
    embedding = torch.stack(
        [
            lift_stabilizer_state(result, v)
            for v in torch.eye(4, dtype=torch.complex128)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        pauli_matrix(full.operator, n_qubits=3) @ embedding,
        embedding @ pauli_matrix(result.operator, n_qubits=2),
    )
    torch.testing.assert_close(
        pauli_matrix(symmetry, n_qubits=3) @ embedding, -embedding
    )


def test_symbolic_large_register_and_zero_sector_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = pauli(factors=((0, "x"), (99, "y")))

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("symbolic reduction allocated a dense tensor")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "zeros", forbidden)
        patch.setattr(torch, "eye", forbidden)
        result = reduce_stabilizers(
            pauli(0.4) + 0.2 * generator,
            n_qubits=100,
            generators=[generator],
            eigenvalues=[1],
        )
    assert result.n_qubits == 99
    assert result.operator.terms[0].coefficient == pytest.approx(0.6)
    x = word("x")
    zero = reduce_stabilizers(pauli() + x, n_qubits=1, generators=[x], eigenvalues=[-1])
    assert zero.n_qubits == 0
    assert not zero.operator.terms
    complex_result = reduce_stabilizers(
        (1 + 2j) * x, n_qubits=1, generators=[x], eigenvalues=[-1]
    )
    assert complex_result.operator == pauli(-1 - 2j)
