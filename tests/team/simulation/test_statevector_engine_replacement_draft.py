"""Draft replacement proof using the existing statevector consumer seam.

This intentionally defines no production protocol.  It records the consumer
behavior that a future Core-owned Simulation Engine contract must preserve.
"""

from dataclasses import dataclass

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR
from flagquantum.runtime.execution import run_native

pytestmark = pytest.mark.integration


@dataclass
class _FakeLocalStatevectorProgram:
    """Test-only engine-backed program accepted by the current Runtime consumer."""

    ir: CircuitIR
    output: torch.Tensor
    calls: int = 0
    device: str = "cpu"

    def to_ir(self) -> CircuitIR:
        return self.ir

    def state(self) -> torch.Tensor:
        self.calls += 1
        return self.output


def test_circuit_state_facade_delegates_to_simulation(monkeypatch):
    import flagquantum.simulation.statevector as statevector

    circuit = fq.Circuit(1)
    expected = torch.tensor([[0.0, 1.0j]], dtype=torch.complex64)
    calls = []

    def replacement(candidate, *, refresh=False):
        calls.append((candidate, refresh))
        return expected

    monkeypatch.setattr(statevector, "state", replacement)

    assert circuit.state(refresh=True) is expected
    assert calls == [(circuit, True)]


def test_runtime_consumer_can_replace_local_statevector_implementation():
    source = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    replacement_output = torch.tensor(
        [[2**-0.5, 0.0, 0.0, -(2**-0.5)]], dtype=torch.complex128
    )
    replacement = _FakeLocalStatevectorProgram(source.to_ir(), replacement_output)

    actual, plan = run_native(
        replacement,
        mode="statevector",
        return_plan=True,
        optimize=False,
        device="cpu",
    )

    assert replacement.calls == 1
    assert actual is replacement_output
    assert plan.state_mode == "statevector"
    assert plan.world_size == 1


def test_replacement_output_retains_parameter_gradient_ownership():
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    source = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=theta)
    output = torch.stack((torch.cos(theta / 2), torch.sin(theta / 2))).reshape(1, 2)
    replacement = _FakeLocalStatevectorProgram(
        source.to_ir(), output.to(torch.complex128)
    )

    actual = run_native(
        replacement,
        mode="statevector",
        optimize=False,
        device="cpu",
        require_gradients=True,
    )
    loss = (torch.abs(actual[:, 0]) ** 2 - torch.abs(actual[:, 1]) ** 2).sum()
    loss.backward()

    assert theta.grad is not None
    assert torch.allclose(
        theta.grad,
        -torch.sin(theta.detach()),
        atol=1e-12,
        rtol=1e-12,
    )
