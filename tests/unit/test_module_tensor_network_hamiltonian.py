from __future__ import annotations

from functools import partial

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.errors import ExecutionError
from flagquantum.simulation.tensor_network import observables
from flagquantum.simulation.tensor_network.state import TensorNetworkState

pytestmark = pytest.mark.unit

INIT = torch.tensor([0.31, -0.77], dtype=torch.float64)
TARGET = 0.4


def build(parameters: torch.Tensor) -> fq.Circuit:
    """Build a three-qubit ansatz with shared and nonlinear parameters."""

    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.ry(0, parameters[0]).rx(1, parameters[1]).ry(1, parameters[1])
    circuit.cx(0, 1).ry(2, parameters[0] * parameters[1]).cx(1, 2)
    return circuit


def build_batch(parameters: torch.Tensor) -> fq.Circuit:
    """Build the same ansatz with two declared batch examples."""

    circuit = fq.Circuit(3, bsz=2, dtype=torch.complex128)
    circuit.ry(0, parameters[0]).rx(1, parameters[1]).ry(1, parameters[1])
    circuit.cx(0, 1).ry(2, parameters[0] * parameters[1]).cx(1, 2)
    return circuit


def build_width(width: int, parameters: torch.Tensor) -> fq.Circuit:
    """Build a two-parameter ladder spanning ``width`` qubits."""

    circuit = fq.Circuit(width, dtype=torch.complex128)
    circuit.ry(0, parameters[0])
    for wire in range(width - 1):
        circuit.cx(wire, wire + 1)
    circuit.rx(width - 1, parameters[1])
    return circuit


def mixed_pauli_sum() -> Hamiltonian:
    """Return a real Pauli sum covering X, Y, Z and a distant pair."""

    return Hamiltonian(
        [
            pauli_term(0.7, "Z", (0,)),
            pauli_term(-0.4, "X", (1,)),
            pauli_term(0.3, "Y", (2,)),
            pauli_term(0.5, "ZZ", (0, 1)),
            pauli_term(0.25, "XX", (0, 2)),
        ]
    )


def module(mode: str, *, builder=build) -> fq.Module:
    """Build a Hamiltonian-energy module for one representation."""

    return fq.Module(
        builder,
        2,
        init=INIT.clone(),
        dtype=torch.float64,
        hamiltonian=mixed_pauli_sum(),
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode=mode,
                precision="complex128",
                allow_backend_fallback=False,
            ),
            observable="hamiltonian",
            observable_qubits=(0, 1, 2),
        ),
    )


def train_step(module_: fq.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """Run one optimizer update on the squared error against ``TARGET``."""

    optimizer.zero_grad()
    loss = (module_() - TARGET) ** 2
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_tensor_network_hamiltonian_matches_statevector_value() -> None:
    energies = {
        mode: module(mode)().detach() for mode in ("statevector", "tensor_network")
    }

    torch.testing.assert_close(
        energies["tensor_network"], energies["statevector"], rtol=1e-12, atol=1e-12
    )


def test_tensor_network_hamiltonian_matches_statevector_gradient() -> None:
    gradients = {}
    losses = {}
    for mode in ("statevector", "tensor_network"):
        module_ = module(mode)
        loss = (module_() - TARGET) ** 2
        loss.backward()
        losses[mode] = loss.detach()
        assert module_.parameters_tensor.grad is not None
        gradients[mode] = module_.parameters_tensor.grad.detach().clone()

    torch.testing.assert_close(
        losses["tensor_network"], losses["statevector"], rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        gradients["tensor_network"],
        gradients["statevector"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_tensor_network_hamiltonian_contracts_without_a_dense_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_dense_state(self: TensorNetworkState) -> torch.Tensor:
        raise AssertionError("Hamiltonian contraction must not materialize a state")

    monkeypatch.setattr(TensorNetworkState, "state", refuse_dense_state)
    monkeypatch.setattr(TensorNetworkState, "to_statevector", refuse_dense_state)

    result = module("tensor_network").execute()
    result.value.sum().backward()

    assert result.runtime["mode"] == "tensor_network"
    assert result.runtime["full_state_materialized"] is False
    assert result.runtime["observable_execution"] == "hamiltonian_mpo_contraction"
    assert isinstance(result.runtime["observable_peak_size"], int)
    assert result.runtime["observable_peak_size"] > 0
    assert result.metrics["observable_count"] == 5


def test_tensor_network_hamiltonian_reports_the_observed_contraction_peak() -> None:
    """The recorded peak must be the profile of the ordering that ran.

    A caller choosing the tensor-network representation is choosing it for the
    peak, so the number has to come from the operator network this call
    contracted, not from the state network and not from a second ordering.
    """

    from flagquantum.simulation.tensor_network.entrypoints import run_tensor_network
    from flagquantum.simulation.tensor_network.observables import (
        build_tensor_network_hamiltonian_expectation,
    )

    hamiltonian = many_term_hamiltonian(4)
    module_ = fq.Module(
        partial(build_width, 4),
        2,
        init=INIT.clone(),
        dtype=torch.float64,
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode="tensor_network",
                precision="complex128",
                allow_backend_fallback=False,
            ),
            observable="hamiltonian",
            observable_qubits=tuple(range(4)),
        ),
    )
    result = module_.execute()

    state = run_tensor_network(build_width(4, INIT), dense_observable_qubits=12)
    reference = build_tensor_network_hamiltonian_expectation(
        state.plan, hamiltonian
    ).contraction_profile("greedy")

    assert result.runtime["observable_peak_size"] == reference.peak_size


def many_term_hamiltonian(width: int) -> Hamiltonian:
    """Return a Pauli sum whose term count is four times the circuit width."""

    return Hamiltonian(
        [pauli_term(-0.5, "Z", (wire,)) for wire in range(width)]
        + [pauli_term(-0.7, "X", (wire,)) for wire in range(width)]
        + [pauli_term(0.1, "ZZ", (wire, wire + 1)) for wire in range(width - 1)]
        + [pauli_term(-0.2, "XX", (wire, wire + 1)) for wire in range(width - 1)]
    )


def test_tensor_network_hamiltonian_contracts_one_network_for_the_whole_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sum must not contract the shared state once per term."""

    builds: list[int] = []
    original_build = observables.build_tensor_network_hamiltonian_expectation

    def counting_build(plan: object, hamiltonian: Hamiltonian) -> object:
        builds.append(int(hamiltonian.n_terms))
        return original_build(plan, hamiltonian)

    def refuse_per_term(self: TensorNetworkState, **_: object) -> torch.Tensor:
        raise AssertionError("a Pauli sum must not contract one term at a time")

    monkeypatch.setattr(
        observables, "build_tensor_network_hamiltonian_expectation", counting_build
    )
    monkeypatch.setattr(TensorNetworkState, "expectation_ps", refuse_per_term)

    for width in (3, 6):
        builds.clear()
        value = fq.Module(
            partial(build_width, width),
            2,
            init=INIT.clone(),
            dtype=torch.float64,
            hamiltonian=many_term_hamiltonian(width),
            policy=fq.RuntimePolicy(
                execution_options=fq.ExecutionOptions(
                    mode="tensor_network",
                    precision="complex128",
                    allow_backend_fallback=False,
                ),
                observable="hamiltonian",
                observable_qubits=tuple(range(width)),
            ),
        )()
        value.sum().backward()
        assert len(builds) == 1, (width, builds)


def test_tensor_network_hamiltonian_agrees_with_statevector_on_many_terms() -> None:
    width = 6

    def builder(parameters: torch.Tensor) -> fq.Circuit:
        return build_width(width, parameters)

    def make(mode: str) -> fq.Module:
        return fq.Module(
            builder,
            2,
            init=INIT.clone(),
            dtype=torch.float64,
            hamiltonian=many_term_hamiltonian(width),
            policy=fq.RuntimePolicy(
                execution_options=fq.ExecutionOptions(
                    mode=mode,
                    precision="complex128",
                    allow_backend_fallback=False,
                ),
                observable="hamiltonian",
                observable_qubits=tuple(range(width)),
            ),
        )

    reference = make("statevector")
    loss = (reference() - TARGET) ** 2
    loss.backward()
    candidate = make("tensor_network")
    other = (candidate() - TARGET) ** 2
    other.backward()

    assert candidate.hamiltonian is not None
    assert candidate.hamiltonian.n_terms == 4 * width - 2
    torch.testing.assert_close(other.detach(), loss.detach(), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        candidate.parameters_tensor.grad.detach(),
        reference.parameters_tensor.grad.detach(),
        rtol=1e-11,
        atol=1e-11,
    )


def test_tensor_network_hamiltonian_refuses_a_qubit_outside_the_circuit() -> None:
    """A term on a qubit the state does not have must refuse, not be dropped."""

    from flagquantum.simulation.tensor_network.entrypoints import run_tensor_network

    state = run_tensor_network(build(INIT), dense_observable_qubits=12)
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", (0,)), pauli_term(5.0, "Z", (7,))])
    with pytest.raises(ValueError, match="observable qubit index out of range"):
        hamiltonian.expectation(state)

    module_ = fq.Module(
        build,
        2,
        init=INIT.clone(),
        dtype=torch.float64,
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode="tensor_network",
                precision="complex128",
                allow_backend_fallback=False,
            ),
            observable="hamiltonian",
            observable_qubits=(0,),
        ),
    )
    with pytest.raises(ExecutionError) as raised:
        module_()
    assert isinstance(raised.value.__cause__, ValueError)
    assert "observable qubit index out of range" in str(raised.value.__cause__)


def test_tensor_network_hamiltonian_includes_a_constant_term() -> None:
    module_ = fq.Module(
        build,
        2,
        init=INIT.clone(),
        dtype=torch.float64,
        hamiltonian=Hamiltonian(
            [
                pauli_term(0.25, "", ()),
                pauli_term(0.7, "Z", (0,)),
                pauli_term(-0.4, "X", (1,)),
            ]
        ),
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode="tensor_network",
                precision="complex128",
                allow_backend_fallback=False,
            ),
            observable="hamiltonian",
            observable_qubits=(0, 1),
        ),
    )
    shift = fq.Module(
        build,
        2,
        init=INIT.clone(),
        dtype=torch.float64,
        hamiltonian=Hamiltonian(
            [pauli_term(0.7, "Z", (0,)), pauli_term(-0.4, "X", (1,))]
        ),
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                mode="tensor_network",
                precision="complex128",
                allow_backend_fallback=False,
            ),
            observable="hamiltonian",
            observable_qubits=(0, 1),
        ),
    )
    torch.testing.assert_close(
        module_().detach(), shift().detach() + 0.25, rtol=1e-12, atol=1e-12
    )


def test_tensor_network_hamiltonian_honours_a_contraction_budget() -> None:
    from flagquantum.simulation.tensor_network.entrypoints import run_tensor_network

    circuit = build_width(4, INIT)
    hamiltonian = many_term_hamiltonian(4)
    unlimited = run_tensor_network(circuit, dense_observable_qubits=12)
    bounded = run_tensor_network(
        circuit, dense_observable_qubits=12, max_intermediate_bytes=1 << 16
    )

    torch.testing.assert_close(
        hamiltonian.expectation(bounded),
        hamiltonian.expectation(unlimited),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bounded.summary()["observable_execution"] == "hamiltonian_mpo_contraction"
    # The unsliced profile would describe a network that did not run.
    assert isinstance(unlimited.summary()["observable_peak_size"], int)
    assert bounded.summary()["observable_peak_size"] is None


def test_tensor_network_hamiltonian_supports_a_declared_batch() -> None:
    single = module("tensor_network")().detach()
    batched = module("tensor_network", builder=build_batch)()

    assert batched.shape == (2,)
    torch.testing.assert_close(batched.detach(), single.expand(2), rtol=0, atol=1e-12)


def test_tensor_network_hamiltonian_trains_and_resumes_from_a_checkpoint(
    tmp_path,
) -> None:
    source = module("tensor_network")
    optimizer = torch.optim.Adam(source.parameters(), lr=0.05)
    for _ in range(2):
        train_step(source, optimizer)
    path = source.save_checkpoint(
        tmp_path / "tn_hamiltonian.pt", optimizer=optimizer, seed=11, step=2
    )

    expected_losses = [train_step(source, optimizer) for _ in range(2)]
    expected_parameters = source.parameters_tensor.detach().clone()

    target = module("tensor_network")
    target_optimizer = torch.optim.Adam(target.parameters(), lr=0.05)
    restored = target.load_checkpoint(path, optimizer=target_optimizer)
    actual_losses = [train_step(target, target_optimizer) for _ in range(2)]

    assert restored["step"] == 2
    torch.testing.assert_close(actual_losses, expected_losses, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        target.parameters_tensor.detach(), expected_parameters, rtol=1e-12, atol=1e-12
    )
