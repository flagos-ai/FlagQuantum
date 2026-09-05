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
from flagquantum.simulation.statevector import run_local_statevector

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


@dataclass
class _RealLocalStatevectorProgram:
    """Test adapter exposing the real Simulation implementation at the same seam."""

    ir: CircuitIR
    batch_size: int
    dtype: torch.dtype
    calls: int = 0
    device: str = "cpu"

    def to_ir(self) -> CircuitIR:
        return self.ir

    def state(self) -> torch.Tensor:
        self.calls += 1
        return run_local_statevector(
            self.ir,
            batch_size=self.batch_size,
            device=torch.device(self.device),
            dtype=self.dtype,
        )


def _program_for(
    implementation: str,
    source: fq.Circuit,
    expected: torch.Tensor,
) -> _RealLocalStatevectorProgram | _FakeLocalStatevectorProgram:
    if implementation == "real":
        return _RealLocalStatevectorProgram(
            source.to_ir(),
            batch_size=source.bsz,
            dtype=source.dtype,
        )
    return _FakeLocalStatevectorProgram(source.to_ir(), expected)


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


@pytest.mark.parametrize("implementation", ("real", "fake"))
def test_runtime_consumer_runs_shared_local_statevector_conformance(
    implementation: str,
):
    source = fq.Circuit(2, bsz=2, dtype=torch.complex128).h(0).cx(0, 1)
    replacement_output = torch.tensor(
        [[2**-0.5, 0.0, 0.0, -(2**-0.5)]], dtype=torch.complex128
    ).expand(2, -1)
    expected = (
        torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex128).expand(
            2, -1
        )
        if implementation == "real"
        else replacement_output
    )
    program = _program_for(implementation, source, expected)

    actual, plan = run_native(
        program,
        mode="statevector",
        return_plan=True,
        optimize=False,
        device="cpu",
        bsz=2,
        dtype=torch.complex128,
    )

    assert program.calls == 1
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert actual.shape == (2, 4)
    assert actual.dtype is torch.complex128
    assert plan.state_mode == "statevector"
    assert plan.world_size == 1


@pytest.mark.parametrize("implementation", ("real", "fake"))
def test_shared_conformance_retains_parameter_gradient_ownership(
    implementation: str,
):
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    source = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=theta)
    output = torch.stack((torch.cos(theta / 2), torch.sin(theta / 2))).reshape(1, 2)
    program = _program_for(implementation, source, output.to(torch.complex128))

    actual = run_native(
        program,
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
