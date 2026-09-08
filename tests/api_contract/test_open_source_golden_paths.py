from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.integration


def test_bell_circuit_build_execute_and_measure() -> None:
    circuit = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
    request = fq.MeasurementNode("probabilities", (0, 1))

    result = fq.run(circuit, measurements=(request,))

    assert result.plan is not None
    assert result.state is not None
    assert torch.allclose(
        result.measurements[0].value,
        torch.tensor([[0.5, 0.0, 0.0, 0.5]]),
        atol=1e-6,
    )


def test_parameterized_variational_objective_trains() -> None:
    def build(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(n_qubits=1).ry(0, theta=parameters[0])

    module = fq.Module(build, n_parameters=1, init=torch.tensor([0.25]))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.2)

    training = fq.train(
        module,
        optimizer=optimizer,
        objective=lambda value: value.mean(),
        steps=3,
    )

    assert training.completed_steps == 3
    assert training.losses[-1] < training.losses[0]
    assert training.last_execution.value is not None


def test_quantum_module_composes_with_pytorch() -> None:
    def build(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(n_qubits=1).ry(0, theta=parameters[0])

    class HybridModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quantum = fq.Module(build, n_parameters=1, init="uniform", seed=7)
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self) -> torch.Tensor:
            return self.scale * self.quantum().mean()

    model = HybridModel()
    loss = model().square()
    loss.backward()

    assert model.scale.grad is not None
    assert next(model.quantum.parameters()).grad is not None


def test_automatic_backend_planning_is_explainable() -> None:
    circuit = fq.Circuit(n_qubits=2).h(0).cx(0, 1)

    plan = fq.plan(circuit)
    summary = plan.summary()

    assert plan.state_mode == "statevector"
    assert plan.recommended_mode == "local"
    assert summary["distribution_semantics"] == "single_device_fast_path"
    assert summary["state_bytes"] == 32


def test_local_deployment_package_executes() -> None:
    circuit = fq.Circuit(n_qubits=2).x(0).x(1)
    provider = InMemoryRemoteTarget()
    backend = provider.discover_backends(2)[0]

    result = fqd.deploy_circuit(circuit, provider, backend=backend, shots=16)

    assert result.handle.provider == "local"
    assert result.counts == {"11": 16}


@pytest.mark.qiskit
def test_qiskit_round_trip_preserves_bell_program() -> None:
    pytest.importorskip("qiskit")
    from flagquantum.ecosystem.qiskit import export_qiskit, import_qiskit

    original = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
    exported = export_qiskit(original)
    restored = import_qiskit(exported.circuit)

    assert exported.report.lossless
    assert restored.report.lossless
    assert restored.ir.instructions == original.to_ir().instructions
