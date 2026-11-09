"""Low-rank dynamics compared to independent Fock-space Hamiltonians."""

import itertools

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)
from flagquantum.algorithms.fermion.evolution import low_rank_evolution
from flagquantum.algorithms.fermion.low_rank import LowRankFactors, decompose_two_body
from flagquantum.algorithms.fermion.swap_network import fermionic_swap_network
from flagquantum.observables import annihilate, create
from flagquantum.observables.fermion import FermionOperator
from flagquantum.simulation.unitary import get_unitary

from .oracles import fock_matrix

pytestmark = pytest.mark.unit


def problem() -> MolecularProblem:
    generator = torch.Generator().manual_seed(109)
    factors = torch.randn(2, 2, 2, generator=generator, dtype=torch.float64)
    factors = (factors + factors.transpose(1, 2)) / 2
    tensor = torch.einsum(
        "l,lpq,lrs->pqrs",
        torch.tensor([0.25, -0.13], dtype=torch.float64),
        factors,
        factors,
    )
    return MolecularProblem(
        torch.tensor([[-0.9, 0.17], [0.17, 0.4]], dtype=torch.float64),
        tensor,
        2,
        0.3,
        -0.12,
    )


def test_low_rank_reconstructs_signed_interactions_and_correction() -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    reconstructed = torch.einsum(
        "l,lpq,lrs->pqrs", factors.eigenvalues, factors.factors, factors.factors
    )
    torch.testing.assert_close(reconstructed, molecule.two_body)
    width = molecule.n_modes
    correction = FermionOperator()
    for p in range(2):
        for q in range(2):
            for spin in range(2):
                correction = correction + float(factors.correction[p, q]) * create(
                    2 * p + spin
                ) * annihilate(2 * q + spin)
    expected = fock_matrix(correction, width)
    for weight, factor in zip(factors.eigenvalues, factors.factors, strict=True):
        density = FermionOperator()
        for p in range(2):
            for q in range(2):
                for spin in range(2):
                    density = density + float(factor[p, q]) * create(
                        2 * p + spin
                    ) * annihilate(2 * q + spin)
        matrix = fock_matrix(density, width)
        expected = expected + 0.5 * weight * matrix @ matrix
    interaction = MolecularProblem(
        torch.zeros_like(molecule.one_body), molecule.two_body, 2
    )
    torch.testing.assert_close(
        expected, fock_matrix(fermion_hamiltonian(interaction), width)
    )
    truncated = decompose_two_body(molecule.two_body, rank=1)
    delta = molecule.two_body - torch.einsum(
        "l,lpq,lrs->pqrs", truncated.eigenvalues, truncated.factors, truncated.factors
    )
    error = fock_matrix(
        fermion_hamiltonian(
            MolecularProblem(torch.zeros_like(molecule.one_body), delta, 2)
        ),
        width,
    )
    assert (
        float(torch.linalg.matrix_norm(error, ord=2))
        <= truncated.discarded_bound + 1e-10
    )
    assert truncated.residual_norm == pytest.approx(
        float(torch.linalg.vector_norm(delta))
    )
    assert decompose_two_body(molecule.two_body, rank=0).factors.shape == (0, 2, 2)


@pytest.mark.parametrize("order", [1, 2])
def test_product_formula_converges_including_global_phase(order: int) -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    time = 0.37
    target = torch.matrix_exp(
        -1j * time * fock_matrix(fermion_hamiltonian(molecule), 4)
    )
    errors = []
    for steps in (1, 2, 4):
        circuit = low_rank_evolution(
            molecule.one_body,
            factors,
            time,
            constant=molecule.energy_offset,
            steps=steps,
            order=order,
        )
        errors.append(float(torch.linalg.matrix_norm(get_unitary(circuit) - target)))
        assert max(len(step.wires) for step in circuit.to_ir().instructions) <= 2
    assert errors[0] / errors[1] > (1.8 if order == 1 else 3.5)
    assert errors[1] / errors[2] > (1.8 if order == 1 else 3.5)


def test_evolution_time_gradient_matches_finite_difference() -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)

    def loss(time: torch.Tensor | float) -> torch.Tensor:
        circuit = fq.Circuit(4, dtype=torch.complex128).x(0).h(2)
        circuit.compose(
            low_rank_evolution(
                molecule.one_body,
                factors,
                time,
                constant=molecule.energy_offset,
                steps=2,
            )
        )
        state = circuit.state().reshape(-1)
        return state[8].real + state[10].imag

    time = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(loss(time), time)[0]
    finite_difference = (loss(0.230001) - loss(0.229999)) / 0.000002
    torch.testing.assert_close(derivative, finite_difference, atol=1e-7, rtol=1e-6)
    assert float(derivative.abs()) > 1e-3


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_fermionic_swap_layout_pair_coverage_and_signs(n: int) -> None:
    initial = tuple(reversed(range(n)))
    network = fermionic_swap_network(n, initial_layout=initial)
    assert {frozenset(pair) for pair in network.interactions} == {
        frozenset(pair) for pair in itertools.combinations(range(n), 2)
    }
    assert len(network.interactions) == n * (n - 1) // 2
    assert network.final_layout == tuple(reversed(initial))
    unitary = get_unitary(network.circuit)
    for index in range(2**n):
        bits = [(index >> (n - i - 1)) & 1 for i in range(n)]
        target = sum(bit << i for i, bit in enumerate(bits))
        particles = sum(bits)
        assert complex(unitary[target, index]) == pytest.approx(
            (-1) ** (particles * (particles - 1) // 2)
        )


def test_evolution_validation() -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    with pytest.raises(ValueError, match="order"):
        low_rank_evolution(molecule.one_body, factors, 0.1, order=3)
    with pytest.raises(ValueError, match="time"):
        low_rank_evolution(molecule.one_body, factors, torch.tensor([0.1]))
    with pytest.raises(ValueError, match="rank"):
        decompose_two_body(molecule.two_body, rank=5)
    with pytest.raises(ValueError, match="symmetry"):
        decompose_two_body(torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2))


@pytest.mark.parametrize("time", [0.0, -0.3])
def test_single_spatial_orbital_evolution_is_exact(time: float) -> None:
    molecule = MolecularProblem(
        torch.tensor([[-0.8]], dtype=torch.float64),
        torch.tensor([[[[0.6]]]], dtype=torch.float64),
        2,
        0.3,
    )
    circuit = low_rank_evolution(
        molecule.one_body,
        decompose_two_body(molecule.two_body),
        time,
        constant=molecule.energy_offset,
        steps=2,
    )
    expected = torch.diag(
        torch.exp(
            -1j * time * torch.tensor([0.3, -0.5, -0.5, -0.7], dtype=torch.complex128)
        )
    )
    torch.testing.assert_close(get_unitary(circuit), expected)


def test_low_rank_factors_reject_detached_or_inconsistent_data() -> None:
    with pytest.raises(ValueError, match="autograd"):
        LowRankFactors(
            torch.ones(1, requires_grad=True),
            torch.ones(1, 1, 1),
            torch.zeros(1, 1),
            0,
            0,
        )
    with pytest.raises(ValueError, match="dimensions"):
        LowRankFactors(torch.ones(2), torch.ones(1, 1, 1), torch.zeros(1, 1), 0, 0)
