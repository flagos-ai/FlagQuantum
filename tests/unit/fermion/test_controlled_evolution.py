"""Controlled dynamics checked against independent Fock-space exponentials."""

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)
from flagquantum.algorithms.fermion.evolution import low_rank_evolution
from flagquantum.algorithms.fermion.low_rank import LowRankFactors, decompose_two_body
from flagquantum.observables import annihilate, create
from flagquantum.observables.fermion import FermionOperator
from flagquantum.simulation.unitary import get_unitary

from .oracles import fock_matrix
from .test_evolution import problem

pytestmark = pytest.mark.unit


def controlled_matrix(unitary: torch.Tensor) -> torch.Tensor:
    """Append the control as the least significant bit, independently of IR."""
    zero = torch.diag(torch.tensor([1, 0], dtype=torch.complex128))
    one = torch.diag(torch.tensor([0, 1], dtype=torch.complex128))
    return torch.kron(
        torch.eye(len(unitary), dtype=torch.complex128), zero
    ) + torch.kron(unitary.contiguous(), one)


def density_matrix(matrix: torch.Tensor) -> torch.Tensor:
    operator = FermionOperator()
    for p in range(len(matrix)):
        for q in range(len(matrix)):
            for spin in range(2):
                operator = operator + complex(matrix[p, q]) * create(
                    2 * p + spin
                ) * annihilate(2 * q + spin)
    return fock_matrix(operator, 2 * len(matrix))


def formula_matrix(
    one_body: torch.Tensor,
    factors: LowRankFactors,
    time: torch.Tensor | float,
    constant: float,
    steps: int,
    order: int,
) -> torch.Tensor:
    """Exponentiate independently assembled density squares, without gate lowering."""
    components = [density_matrix(one_body + factors.correction)]
    for weight, factor in zip(factors.eigenvalues, factors.factors, strict=True):
        density = density_matrix(factor)
        components.append(0.5 * weight * density @ density)
    schedule = (
        [(component, 1.0) for component in components]
        if order == 1
        else (
            [(component, 0.5) for component in components]
            + [(component, 0.5) for component in reversed(components)]
        )
    )
    result = torch.exp(
        -1j * torch.as_tensor(time, dtype=torch.float64) * constant
    ) * torch.eye(1 << (2 * len(one_body)), dtype=torch.complex128)
    for _ in range(steps):
        for matrix, fraction in schedule:
            result = torch.matrix_exp(-1j * time * fraction / steps * matrix) @ result
    return result


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("time", [0.0, -0.23, 0.37])
def test_complete_control_branches_and_product_formula(order: int, time: float) -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    block = low_rank_evolution(
        molecule.one_body,
        factors,
        time,
        constant=molecule.energy_offset,
        steps=2,
        order=order,
    ).control(1, ctrl_qubits=4)
    expected = formula_matrix(
        molecule.one_body, factors, time, molecule.energy_offset, 2, order
    )
    actual = get_unitary(block)
    torch.testing.assert_close(
        actual, controlled_matrix(expected), atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        actual[::2, ::2], torch.eye(16, dtype=torch.complex128), atol=2e-12, rtol=0
    )
    assert not actual[::2, 1::2].abs().any()
    assert not actual[1::2, ::2].abs().any()


@pytest.mark.parametrize("order", [1, 2])
def test_controlled_evolution_converges_to_physical_hamiltonian(order: int) -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    target = controlled_matrix(
        torch.matrix_exp(-0.37j * fock_matrix(fermion_hamiltonian(molecule), 4))
    )
    errors = []
    for steps in (1, 2, 4):
        block = low_rank_evolution(
            molecule.one_body,
            factors,
            0.37,
            constant=molecule.energy_offset,
            steps=steps,
            order=order,
        ).control(1, ctrl_qubits=4)
        errors.append(float(torch.linalg.matrix_norm(get_unitary(block) - target)))
    assert errors[0] / errors[1] > (1.8 if order == 1 else 3.5)
    assert errors[1] / errors[2] > (1.8 if order == 1 else 3.5)


@pytest.mark.parametrize("rank", [0, 1])
def test_truncated_factors_keep_original_contraction(rank: int) -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body, rank=rank)
    block = low_rank_evolution(
        molecule.one_body, factors, 0.4, constant=molecule.energy_offset, steps=2
    ).control(1, ctrl_qubits=4)
    expected = formula_matrix(
        molecule.one_body, factors, 0.4, molecule.energy_offset, 2, 2
    )
    torch.testing.assert_close(
        get_unitary(block), controlled_matrix(expected), atol=2e-12, rtol=2e-12
    )
    assert factors.discarded_bound > 0


def test_complex_one_body_basis_and_zero_weight_factor() -> None:
    matrix = torch.tensor([[0.3, 0.2j], [-0.2j, -0.4]], dtype=torch.complex128)
    factors = LowRankFactors(
        torch.zeros(1), torch.zeros(1, 2, 2), torch.zeros(2, 2), 0, 0
    )
    block = low_rank_evolution(matrix, factors, -0.31, constant=0.7).control(
        1, ctrl_qubits=4
    )
    target = torch.matrix_exp(
        0.31j * (density_matrix(matrix) + 0.7 * torch.eye(16, dtype=torch.complex128))
    )
    torch.testing.assert_close(
        get_unitary(block), controlled_matrix(target), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("offset", [-1.7, 0.0, 0.8])
def test_constant_phase_is_measurable_on_superposed_control(offset: float) -> None:
    factors = LowRankFactors(
        torch.empty(0), torch.empty(0, 1, 1), torch.zeros(1, 1), 0, 0
    )
    time = 0.37
    block = low_rank_evolution(
        torch.zeros(1, 1), factors, time, constant=offset
    ).control(1, ctrl_qubits=2)
    circuit = fq.Circuit(3, dtype=torch.complex128).h(2)
    circuit.compose(block)
    values = fq.run(circuit, outputs=[fq.expectation(fq.X(2)), fq.expectation(fq.Y(2))])
    x, y = [torch.as_tensor(value).reshape(()) for value in values.expectations]
    phase = torch.tensor(time * offset, dtype=torch.float64)
    torch.testing.assert_close(x, torch.cos(phase))
    torch.testing.assert_close(y, -torch.sin(phase))
    differentiable_time = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    differentiable = fq.Circuit(3, dtype=torch.complex128).h(2)
    differentiable.compose(
        low_rank_evolution(
            torch.zeros(1, 1), factors, differentiable_time, constant=offset
        ).control(1, ctrl_qubits=2)
    )
    state = differentiable.state().reshape(-1)
    coherence = 2 * (state[::2].conj() * state[1::2]).sum()
    derivative = torch.autograd.grad(
        coherence.real + coherence.imag, differentiable_time
    )[0]
    torch.testing.assert_close(
        derivative,
        -offset
        * (
            torch.sin(differentiable_time * offset)
            + torch.cos(differentiable_time * offset)
        ),
    )


@pytest.mark.parametrize("order", [1, 2])
def test_time_gradient_matches_independent_exponentials(order: int) -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    time = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[24:26] = 2**-0.5

    def loss(value: torch.Tensor | float) -> torch.Tensor:
        circuit = fq.Circuit(5, dtype=torch.complex128, inputs=initial)
        circuit.compose(
            low_rank_evolution(
                molecule.one_body,
                factors,
                value,
                constant=molecule.energy_offset,
                steps=2,
                order=order,
            ).control(1, ctrl_qubits=4)
        )
        state = circuit.state().reshape(-1)
        return torch.as_tensor(2 * (state[::2].conj() * state[1::2]).sum().real)

    derivative = torch.autograd.grad(loss(time), time)[0]
    reference = formula_matrix(
        molecule.one_body, factors, time, molecule.energy_offset, 2, order
    )[12, 12].real
    expected = torch.autograd.grad(reference, time)[0]
    finite = (loss(0.230001) - loss(0.229999)) / 0.000002
    torch.testing.assert_close(derivative, expected, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(derivative, finite, atol=1e-8, rtol=1e-7)
    assert derivative.abs() > 1e-3


def test_single_orbital_exact_all_particle_sectors() -> None:
    molecule = MolecularProblem(
        torch.tensor([[-0.8]]), torch.tensor([[[[0.6]]]]), 2, 0.3
    )
    factors = decompose_two_body(molecule.two_body)
    expected = torch.matrix_exp(0.3j * fock_matrix(fermion_hamiltonian(molecule), 2))
    block = low_rank_evolution(molecule.one_body, factors, -0.3, constant=0.3).control(
        1, ctrl_qubits=2
    )
    torch.testing.assert_close(get_unitary(block), controlled_matrix(expected))


def test_synthesis_resources_grow_with_steps_not_full_hilbert_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_exponential(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthesis must not exponentiate a full-system matrix")

    monkeypatch.setattr(torch, "matrix_exp", no_exponential)
    n = 8
    factors = LowRankFactors(
        torch.tensor([-0.2]),
        torch.diag(torch.arange(n, dtype=torch.float64)).unsqueeze(0),
        torch.zeros(n, n),
        0,
        0,
    )
    lengths = []
    for steps in (1, 2, 4):
        base = low_rank_evolution(
            torch.diag(torch.arange(n, dtype=torch.float64)),
            factors,
            0.2,
            constant=0.4,
            steps=steps,
        )
        controlled = base.control(1, ctrl_qubits=2 * n)
        source, target = base.to_ir().instructions, controlled.to_ir().instructions
        assert len(source) == len(target)
        assert controlled.n_qubits == 2 * n + 1
        assert all(
            len(gate.wires) <= 3
            and gate.matrix is not None
            and gate.matrix.shape[0] <= 8
            for gate in target
        )
        lengths.append(len(target))
    assert lengths[1] - 1 == 2 * (lengths[0] - 1)
    assert lengths[2] - 1 == 4 * (lengths[0] - 1)


def test_static_autograd_inputs_are_refused() -> None:
    molecule = problem()
    factors = decompose_two_body(molecule.two_body)
    with pytest.raises(ValueError, match="autograd"):
        low_rank_evolution(
            molecule.one_body.clone().requires_grad_(), factors, 0.2
        ).control(1, ctrl_qubits=4)
