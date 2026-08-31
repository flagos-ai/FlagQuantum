"""Tests for native FlagQuantum algorithm utilities."""

import torch

import flagquantum as fq
import flagquantum.backends as fqb


def test_hamiltonian_expectation_on_bell_circuit_and_mps():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(0.5, "ZZ", (0, 1)),
            fq.pauli_term(0.25, "XX", (0, 1)),
        ]
    )

    mps = fqb.run_mps(circuit)

    assert torch.allclose(hamiltonian.expectation(circuit), torch.tensor([0.75]))
    assert torch.allclose(hamiltonian.expectation(mps), torch.tensor([0.75]))


def test_hamiltonian_expectation_on_statevector_and_density_matrix():
    circuit = fq.Circuit(1)
    circuit.x(0)
    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(2.0, "Z", (0,)),
            fq.pauli_term(0.5, "I", (0,)),
        ]
    )

    state_value = hamiltonian.expectation(circuit.state())
    density_value = hamiltonian.expectation(circuit.density_matrix())

    assert torch.allclose(state_value, torch.tensor([-1.5]))
    assert torch.allclose(density_value, torch.tensor([-1.5]))


def test_mps_z_zz_chain_fastpath_matches_termwise_energy_and_gradient():
    parameters = torch.linspace(0.1, 0.6, 6, requires_grad=True)
    reference_parameters = parameters.detach().clone().requires_grad_(True)
    hamiltonian = fq.zz_chain_hamiltonian(6, coupling=-0.7, field=0.2)

    def build(values):
        circuit = fq.Circuit(6)
        for wire in range(6):
            circuit.ry(wire, values[wire])
        return circuit.cx(0, 1).cx(2, 3).cx(4, 5)

    state = fqb.run_mps(build(parameters), max_bond=4, dense_observable_wires=0)
    reference_state = fqb.run_mps(
        build(reference_parameters), max_bond=4, dense_observable_wires=0
    )
    actual = hamiltonian.expectation(state).sum()
    reference = sum(
        term.expectation(reference_state).sum() for term in hamiltonian.terms
    )
    actual_gradient = torch.autograd.grad(actual, parameters)[0]
    reference_gradient = torch.autograd.grad(reference, reference_parameters)[0]

    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        actual_gradient, reference_gradient, atol=1e-6, rtol=1e-6
    )


def test_vqe_loss_is_differentiable():
    theta = torch.tensor([0.3], requires_grad=True)
    hamiltonian = fq.Hamiltonian([fq.pauli_term(1.0, "Z", (0,))])

    def builder(params):
        circuit = fq.Circuit(1)
        circuit.rx(0, theta=params[0])
        return circuit

    loss = fq.vqe_loss(builder, theta, hamiltonian)
    loss.backward()

    assert torch.allclose(loss.detach(), torch.cos(theta.detach()), atol=1e-6)
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


def test_hardware_efficient_ansatz_parameter_count_and_state_norm():
    count = fq.hardware_efficient_parameter_count(3, 2)
    params = torch.linspace(0.0, 0.5, count)

    circuit = fq.hardware_efficient_ansatz(3, 2, params)
    state = circuit.state()

    assert count == 12
    assert len(circuit) == 16
    assert torch.allclose(
        torch.sum(torch.abs(state) ** 2, dim=-1), torch.ones(1), atol=1e-6
    )


def test_qaoa_circuit_builds_weighted_cost_layers():
    gammas = torch.tensor([0.2], requires_grad=True)
    betas = torch.tensor([0.3], requires_grad=True)
    circuit = fq.qaoa_circuit(3, [(0, 1), (1, 2, 0.5)], gammas, betas)
    hamiltonian = fq.zz_chain_hamiltonian(3)

    value = hamiltonian.expectation(circuit).sum()
    value.backward()

    assert len(circuit) == 8
    assert gammas.grad is not None
    assert betas.grad is not None
    assert torch.allclose(
        torch.sum(torch.abs(circuit.state()) ** 2, dim=-1),
        torch.ones(1),
        atol=1e-6,
    )


def test_qaoa_loss_is_differentiable():
    gammas = torch.tensor([0.2], requires_grad=True)
    betas = torch.tensor([0.3], requires_grad=True)
    hamiltonian = fq.zz_chain_hamiltonian(2)

    loss = fq.qaoa_loss(2, [(0, 1)], gammas, betas, hamiltonian)
    loss.backward()

    assert gammas.grad is not None
    assert betas.grad is not None


def test_run_vqe_reduces_energy():
    hamiltonian = fq.Hamiltonian([fq.pauli_term(1.0, "Z", (0,))])

    def builder(params):
        circuit = fq.Circuit(1)
        circuit.rx(0, theta=params[0])
        return circuit

    initial = torch.tensor([1.0])
    initial_energy = fq.vqe_loss(builder, initial, hamiltonian).detach()
    result = fq.run_vqe(builder, initial, hamiltonian, steps=25, lr=0.2)

    assert result.n_steps == 25
    assert result.parameters.shape == initial.shape
    assert result.energy < initial_energy
    assert result.history[-1] < result.history[0]


def test_run_adapt_vqe_selects_largest_exact_gradient_and_reduces_energy():
    hamiltonian = fq.Hamiltonian([fq.pauli_term(1.0, "Z", (0,))])
    pool = ("rx", "ry")

    def builder(operators, parameters):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        circuit.h(0)
        for operator, parameter in zip(operators, parameters):
            getattr(circuit, operator)(0, theta=parameter)
        return circuit

    result = fq.run_adapt_vqe(
        builder,
        pool,
        hamiltonian,
        max_adapt_iterations=1,
        optimization_steps=50,
        lr=0.1,
    )

    assert result.selected_pool_indices == (1,)
    assert result.n_adapt_iterations == 1
    assert abs(result.iterations[0].pool_gradients[0]) < 1e-12
    assert abs(result.iterations[0].pool_gradients[1]) > 0.99
    assert result.energy < -0.98
    assert float(result.energy) < result.initial_energy


def test_run_adapt_vqe_accepts_tensor_network_energy_evaluator():
    pool = ("rx", "ry")

    def builder(operators, parameters):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        circuit.h(0)
        for operator, parameter in zip(operators, parameters):
            getattr(circuit, operator)(0, theta=parameter)
        return circuit

    def tn_energy(circuit):
        state = fqb.run_native(circuit, mode="tensor_network")
        return state.expectation_ps(z=(0,))

    result = fq.run_adapt_vqe(
        builder,
        pool,
        energy_function=tn_energy,
        max_adapt_iterations=1,
        optimization_steps=20,
        lr=0.1,
    )

    assert result.selected_pool_indices == (1,)
    assert result.energy < -0.8


def test_run_adapt_vqe_accepts_exact_screening_function():
    pool = (("rx", 0), ("ry", 0))
    target = fq.Hamiltonian([fq.pauli_term(-1.0, "X", (0,))])
    calls = []

    def builder(operators, parameters):
        circuit = fq.Circuit(1, dtype=torch.complex128)
        for (kind, wire), parameter in zip(operators, parameters):
            getattr(circuit, kind)(wire, theta=parameter)
        return circuit

    def screen(operators, parameters, candidates):
        calls.append((operators, tuple(candidates)))
        return (0.0, -1.0)

    result = fq.run_adapt_vqe(
        builder,
        pool,
        target,
        screening_function=screen,
        max_adapt_iterations=1,
        optimization_steps=2,
        dtype=torch.float64,
    )

    assert calls == [((), pool)]
    assert result.selected_pool_indices == (1,)
    assert result.iterations[0].pool_gradients == (0.0, -1.0)
