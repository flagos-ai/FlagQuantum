from __future__ import annotations

import time

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.observability.acceptance import HybridAcceptanceReport

pytestmark = pytest.mark.unit

INPUTS = torch.tensor([[-1.0, -0.5], [-0.7, 0.8], [0.6, -0.9], [0.9, 0.7]])
TARGETS = torch.tensor([-1.0, -1.0, 1.0, 1.0])


def train_classifier(
    model: fq.HybridQuantumClassifier, optimizer: torch.optim.Optimizer, steps: int
) -> tuple[float, float]:
    first = 0.0
    for index in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(INPUTS), TARGETS)
        if index == 0:
            first = float(loss.detach())
        loss.backward()
        optimizer.step()
    return first, float(loss.detach())


def test_classifier_local_training_evaluation_checkpoint_and_deployment(
    tmp_path,
) -> None:
    seed = fq.seed_everything(48)
    model = fq.HybridQuantumClassifier(
        deployment_binding={"provider": "local", "target": "simulator"}
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    started = time.perf_counter()
    first, final = train_classifier(model, optimizer, 20)
    elapsed = time.perf_counter() - started
    predictions = torch.sign(model(INPUTS).detach())
    accuracy = float((predictions == TARGETS).float().mean())
    assert final < first and accuracy >= 0.75

    path = fq.save_training_checkpoint(
        tmp_path / "classifier.pt",
        module=model,
        optimizer=optimizer,
        seed=seed,
        precision=model.quantum.precision,
        step=20,
    )
    restored = fq.HybridQuantumClassifier(
        deployment_binding={"provider": "local", "target": "simulator"}
    )
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.08)
    metadata = fq.load_training_checkpoint(
        path,
        module=restored,
        optimizer=restored_optimizer,
        precision=restored.quantum.precision,
    )
    torch.testing.assert_close(restored(INPUTS), model(INPUTS))
    assert metadata["step"] == 20
    assert restored.deployment_parameters()["binding"]["provider"] == "local"
    for candidate, candidate_optimizer in (
        (model, optimizer),
        (restored, restored_optimizer),
    ):
        candidate_optimizer.zero_grad()
        continued_loss = torch.nn.functional.mse_loss(candidate(INPUTS), TARGETS)
        continued_loss.backward()
        candidate_optimizer.step()
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
    assert (
        optimizer.state_dict()["state"].keys()
        == restored_optimizer.state_dict()["state"].keys()
    )
    report = HybridAcceptanceReport(
        model="classifier",
        policy=model.quantum.policy.__dict__,
        correctness={"checkpoint_parity": True, "loss_decreased": final < first},
        accuracy={"classification_accuracy": accuracy},
        performance={"elapsed_seconds": elapsed, "steps": 20},
        deployment=model.deployment_parameters()["binding"],
    )
    assert report.summary()["accuracy_is_performance_claim"] is False


def test_classifier_deployment_binds_encoded_inputs_to_final_quantum_ir() -> None:
    model = fq.HybridQuantumClassifier(
        deployment_binding={"provider": "local", "target": "simulator"}
    )
    payload = model.deployment_parameters(INPUTS[:2])
    expected_angles = model.encoder(INPUTS[:2]) + model.quantum.parameters_tensor
    torch.testing.assert_close(payload["bound_quantum_parameters"], expected_angles)
    assert payload["binding_requires_inputs"] is False
    assert len(payload["bound_ir"]) == 2
    for ir, angles in zip(payload["bound_ir"], expected_angles):
        assert ir == fq.Circuit(2).ry(0, angles[0]).cx(0, 1).ry(1, angles[1]).to_ir()
        assert all(
            not value.requires_grad
            for instruction in ir.instructions
            for value in instruction.params.values()
            if isinstance(value, torch.Tensor)
        )


def test_variational_energy_same_model_switches_native_and_jax_policy() -> None:
    pytest.importorskip("jax", reason="JAX is an optional backend")

    native = fq.VariationalEnergyModel()
    native.quantum.parameters_tensor.data.copy_(torch.tensor([0.21, -0.32]))
    native_value = native()
    native_gradient = torch.autograd.grad(
        native_value, native.quantum.parameters_tensor
    )[0]

    jax = fq.VariationalEnergyModel(
        policy=fq.RuntimePolicy(
            backend="jax",
            observable="hamiltonian",
            observable_wires=(0, 1),
            allow_backend_fallback=False,
        )
    )
    jax.quantum.parameters_tensor.data.copy_(native.quantum.parameters_tensor.data)
    jax_value = jax()
    jax_gradient = torch.autograd.grad(jax_value, jax.quantum.parameters_tensor)[0]
    torch.testing.assert_close(jax_value, native_value, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(jax_gradient, native_gradient, atol=3e-5, rtol=3e-5)
    result = jax.quantum.execute()
    assert result.compatibility["selected_backend"] == "jax"


def test_classifier_policy_switch_does_not_change_model_class() -> None:
    model = fq.HybridQuantumClassifier()
    assert model.quantum.policy.mode == "statevector"
    model.set_runtime_policy(
        fq.RuntimePolicy(mode="distributed_statevector", observable_wires=(1,))
    )
    assert isinstance(model, fq.HybridQuantumClassifier)
    assert model.quantum.policy.mode == "distributed_statevector"


def test_hybrid_policy_switch_synchronizes_jax_cache_and_debug_hook() -> None:
    model = fq.HybridQuantumClassifier()
    model.quantum._jax_kernel = object()
    model.quantum._jax_kernel_signature = ("stale",)
    model.set_runtime_policy(fq.RuntimePolicy(correctness_debug=True))
    assert model.quantum._jax_kernel is None
    assert model.quantum._jax_kernel_signature is None
    assert model.quantum._correctness_debug_hook_handle is not None
    model.set_runtime_policy(fq.RuntimePolicy(correctness_debug=False))
    assert model.quantum._correctness_debug_hook_handle is None
