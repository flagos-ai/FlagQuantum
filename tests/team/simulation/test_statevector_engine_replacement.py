"""Replacement proof for the local statevector Simulation boundary."""

from dataclasses import dataclass, field

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.local as statevector
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.integration


@dataclass
class _FakeStatevectorEngine:
    """Small test double with the real numerical entry point's call shape."""

    output: torch.Tensor
    calls: list[tuple[CircuitIR, int, torch.device, torch.dtype]] = field(
        default_factory=list
    )

    def __call__(
        self,
        program: CircuitIR,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self.calls.append((program, batch_size, device, dtype))
        return self.output.to(device=device, dtype=dtype).expand(batch_size, -1)


def _options(*, batch_size: int = 1, require_gradients: bool = False):
    return fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        batch_size=batch_size,
        precision="complex128",
        require_gradients=require_gradients,
        allow_backend_fallback=False,
    )


def test_circuit_state_facade_delegates_to_simulation(monkeypatch):
    circuit = fq.Circuit(1)
    expected = torch.tensor([[0.0, 1.0j]], dtype=torch.complex64)
    calls = []

    def replacement(candidate, *, refresh=False):
        calls.append((candidate, refresh))
        return expected

    monkeypatch.setattr(statevector, "state", replacement)

    assert circuit.state(refresh=True) is expected
    assert calls == [(circuit, True)]


def test_runtime_accepts_real_and_replacement_statevector_engines(monkeypatch):
    circuit = fq.Circuit(2, bsz=2, dtype=torch.complex128).h(0).cx(0, 1)
    plan = fq.plan(circuit, options=_options(batch_size=2))
    expected = torch.tensor(
        [[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex128
    ).expand(2, -1)

    real_result = fq.run(plan)
    fake = _FakeStatevectorEngine(expected[:1])
    monkeypatch.setattr(statevector, "run_local_statevector", fake)
    replacement_result = fq.run(plan)

    for result in (real_result, replacement_result):
        assert result.plan is plan
        assert result.state is not None
        assert result.state.shape == (2, 4)
        assert result.state.dtype is torch.complex128
        assert result.state.device.type == "cpu"
        torch.testing.assert_close(result.state, expected, atol=1e-12, rtol=0)
        assert result.runtime["execution_path"] == "local_statevector"
        assert result.runtime["simulation_engine"] == "pytorch_statevector"

    assert len(fake.calls) == 1
    program, batch_size, device, dtype = fake.calls[0]
    assert isinstance(program, CircuitIR)
    assert batch_size == 2
    assert device == torch.device("cpu")
    assert dtype is torch.complex128


@pytest.mark.parametrize("replace_engine", (False, True), ids=("real", "replacement"))
def test_statevector_engine_replacement_preserves_gradients(
    monkeypatch,
    replace_engine: bool,
):
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=theta)
    plan = fq.plan(circuit, options=_options(require_gradients=True))
    if replace_engine:
        output = torch.stack((torch.cos(theta / 2), torch.sin(theta / 2))).reshape(1, 2)
        monkeypatch.setattr(
            statevector,
            "run_local_statevector",
            _FakeStatevectorEngine(output),
        )

    result = fq.run(plan)
    assert result.state is not None
    loss = (
        torch.abs(result.state[:, 0]) ** 2 - torch.abs(result.state[:, 1]) ** 2
    ).sum()
    loss.backward()

    assert theta.grad is not None
    torch.testing.assert_close(
        theta.grad,
        -torch.sin(theta.detach()),
        atol=1e-12,
        rtol=1e-12,
    )
