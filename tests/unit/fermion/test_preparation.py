"""State preparation against minors and independent Fock-space exponentials."""

import itertools

import pytest
import torch

from flagquantum.algorithms.fermion.decomposition import decompose_orbital_rotation
from flagquantum.algorithms.fermion.lowering import append_orbital_rotation
from flagquantum.algorithms.fermion.preparation import prepare_gaussian, prepare_slater
from flagquantum.circuit import Circuit
from flagquantum.observables import annihilate, create, number
from flagquantum.observables.fermion import FermionOperator
from flagquantum.simulation.unitary import get_unitary

from .oracles import fock_matrix

pytestmark = pytest.mark.unit


def random_unitary(n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(711 + n)
    matrix = torch.randn(n, n, generator=generator, dtype=torch.complex128)
    return torch.linalg.qr(matrix)[0]


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_orbital_rotation_on_every_particle_sector(n: int) -> None:
    unitary = random_unitary(n)
    decomposition = decompose_orbital_rotation(unitary)
    reconstructed = torch.diag(decomposition.phases)
    for step in decomposition.rotations:
        reconstructed[step.left : step.left + 2] = (
            step.matrix @ reconstructed[step.left : step.left + 2].clone()
        )
    torch.testing.assert_close(reconstructed, unitary)
    circuit = Circuit(n, dtype=torch.complex128)
    append_orbital_rotation(circuit, unitary)
    actual = get_unitary(circuit)
    expected = torch.zeros_like(actual)
    for column in range(2**n):
        occupied = [i for i in range(n) if column & (1 << (n - i - 1))]
        for rows in itertools.combinations(range(n), len(occupied)):
            index = sum(1 << (n - i - 1) for i in rows)
            expected[index, column] = torch.linalg.det(unitary[list(rows)][:, occupied])
    torch.testing.assert_close(actual, expected)
    assert (
        max((len(step.wires) for step in circuit.to_ir().instructions), default=0) <= 2
    )


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_slater_amplitudes_are_complex_minors(k: int) -> None:
    orbitals = random_unitary(4)[:, :k]
    state = prepare_slater(orbitals).state().reshape(-1)
    expected = torch.zeros_like(state)
    for rows in itertools.combinations(range(4), k):
        index = sum(1 << (3 - i) for i in rows)
        expected[index] = torch.linalg.det(orbitals[list(rows)])
    torch.testing.assert_close(state, expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("odd", [False, True])
def test_gaussian_vacuum_against_dense_quadratic_evolution(n: int, odd: bool) -> None:
    generator = torch.Generator().manual_seed(219 + n)
    operator = FermionOperator()
    for i in range(n):
        operator = operator + (0.13 + i / 10) * number(i)
        for j in range(i + 1, n):
            hopping, pairing = torch.randn(
                2, generator=generator, dtype=torch.complex128
            ).tolist()
            term = hopping * create(i) * annihilate(j) + pairing * create(i) * create(j)
            operator = operator + term + term.dagger()
    evolution = torch.matrix_exp(-0.31j * fock_matrix(operator, n))
    if odd:
        # A last-mode particle-hole transformation changes Gaussian parity.
        x = torch.kron(
            torch.eye(2 ** (n - 1), dtype=torch.complex128),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        )
        evolution = evolution @ x
    ladders = [fock_matrix(annihilate(i), n) for i in range(n)]
    a = torch.empty(n, n, dtype=torch.complex128)
    b = torch.empty_like(a)
    for i, ladder in enumerate(ladders):
        transformed = evolution @ ladder @ evolution.mH
        for j, reference in enumerate(ladders):
            a[i, j] = torch.trace(reference.mH @ transformed) / (2 ** (n - 1))
            b[i, j] = torch.trace(reference @ transformed) / (2 ** (n - 1))
    actual = prepare_gaussian(a, b).state().reshape(-1)
    target = evolution[:, 0]
    assert float(torch.vdot(target, actual).abs()) == pytest.approx(1, abs=1e-10)
    for i in range(n):
        annihilator = sum(
            (a[i, j] * ladders[j] + b[i, j] * ladders[j].mH for j in range(n)),
            torch.zeros_like(evolution),
        )
        torch.testing.assert_close(
            annihilator @ actual, torch.zeros_like(actual), atol=1e-10, rtol=1e-10
        )


def test_static_preparation_rejections() -> None:
    with pytest.raises(ValueError, match="autograd"):
        prepare_slater(torch.eye(2, requires_grad=True))
    with pytest.raises(ValueError, match="orthonormal"):
        prepare_slater(torch.ones(2, 2))
    with pytest.raises(ValueError, match="canonical"):
        prepare_gaussian(torch.eye(2), torch.eye(2))
    with pytest.raises(ValueError, match="unitary"):
        decompose_orbital_rotation(torch.ones(2, 2))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_gaussian_singular_blocks_and_occupied_vacua(n: int) -> None:
    orbitals = random_unitary(n)
    row_mixing = orbitals.T
    for occupied in (tuple(range(n)), tuple(range(0, n, 2))):
        mask = torch.tensor(
            [int(i in occupied) for i in range(n)], dtype=torch.complex128
        )
        a = row_mixing @ torch.diag(1 - mask) @ orbitals.mH
        b = row_mixing @ torch.diag(mask) @ orbitals.T
        actual = prepare_gaussian(a, b).state().reshape(-1)
        target = torch.zeros_like(actual)
        for rows in itertools.combinations(range(n), len(occupied)):
            index = sum(1 << (n - i - 1) for i in rows)
            target[index] = torch.linalg.det(orbitals[list(rows)][:, list(occupied)])
        assert float(torch.vdot(target, actual).abs()) == pytest.approx(1, abs=1e-10)
