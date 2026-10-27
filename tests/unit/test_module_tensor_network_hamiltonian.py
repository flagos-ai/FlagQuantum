from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.simulation.tensor_network.state import TensorNetworkState

pytestmark = pytest.mark.unit

INIT = torch.tensor([0.31, -0.77], dtype=torch.float64)
TARGET = 0.4


def build(parameters: torch.Tensor) -> fq.Circuit:
    """Build a three-wire ansatz with shared and nonlinear parameters."""

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
    assert result.runtime["observable_execution"] == "direct_expectation_contraction"
    assert result.runtime["observable_peak_size"] is None
    assert result.metrics["observable_count"] == 5


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
