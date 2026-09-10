from __future__ import annotations

import time

import pytest
import torch

import flagquantum as fq
import flagquantum.training as fqt
from flagquantum.models import HybridQuantumClassifier, VariationalEnergyModel
from flagquantum.runtime.observability.acceptance import HybridAcceptanceReport
from flagquantum.runtime.training_state import (
    load_training_checkpoint,
    save_training_checkpoint,
)

pytestmark = pytest.mark.unit

INPUTS = torch.tensor([[-1.0, -0.5], [-0.7, 0.8], [0.6, -0.9], [0.9, 0.7]])
TARGETS = torch.tensor([-1.0, -1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "model_type", [HybridQuantumClassifier, VariationalEnergyModel]
)
def test_deployment_rejects_missing_quantum_parameter_tensor(
    model_type: type[HybridQuantumClassifier] | type[VariationalEnergyModel],
) -> None:
    model = model_type()
    model.quantum.register_parameter("parameters_tensor", None)
    with pytest.raises(ValueError, match="requires a quantum parameter tensor"):
        model.deployment_parameters()


@pytest.mark.parametrize(
    "model_type", [HybridQuantumClassifier, VariationalEnergyModel]
)
def test_deployment_quantum_parameters_are_detached_and_independent(
    model_type: type[HybridQuantumClassifier] | type[VariationalEnergyModel],
) -> None:
    model = model_type()
    parameters = model.quantum.parameters_tensor
    assert parameters is not None
    before = parameters.detach().clone()
    exported = model.deployment_parameters()["quantum_parameters"]
    assert not exported.requires_grad
    torch.testing.assert_close(exported, before)
    exported.add_(1)
    torch.testing.assert_close(parameters, before)


def train_classifier(
    model: HybridQuantumClassifier, optimizer: torch.optim.Optimizer, steps: int
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
    seed = fqt.seed_everything(48)
    model = HybridQuantumClassifier(
        deployment_binding={"provider": "local", "target": "simulator"}
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    started = time.perf_counter()
    first, final = train_classifier(model, optimizer, 20)
    elapsed = time.perf_counter() - started
    predictions = torch.sign(model(INPUTS).detach())
    accuracy = float((predictions == TARGETS).float().mean())
    assert final < first and accuracy >= 0.75

    path = save_training_checkpoint(
        tmp_path / "classifier.pt",
        module=model,
        optimizer=optimizer,
        seed=seed,
        precision=model.quantum.precision,
        step=20,
    )
    restored = HybridQuantumClassifier(
        deployment_binding={"provider": "different", "target": "discarded"}
    )
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.08)
    metadata = load_training_checkpoint(
        path,
        module=restored,
        optimizer=restored_optimizer,
        precision=restored.quantum.precision,
    )
    torch.testing.assert_close(restored(INPUTS), model(INPUTS))
    assert metadata["step"] == 20
    assert restored.deployment_parameters()["binding"]["provider"] == "local"
    assert not hasattr(restored.quantum, "deployment_binding")
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
    model = HybridQuantumClassifier(
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

    native = VariationalEnergyModel()
    native.quantum.parameters_tensor.data.copy_(torch.tensor([0.21, -0.32]))
    native_value = native()
    native_gradient = torch.autograd.grad(
        native_value, native.quantum.parameters_tensor
    )[0]

    jax = VariationalEnergyModel(
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(
                backend="jax", allow_backend_fallback=False
            ),
            observable="hamiltonian",
            observable_wires=(0, 1),
        )
    )
    jax.quantum.parameters_tensor.data.copy_(native.quantum.parameters_tensor.data)
    jax_value = jax()
    jax_gradient = torch.autograd.grad(jax_value, jax.quantum.parameters_tensor)[0]
    torch.testing.assert_close(jax_value, native_value, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(jax_gradient, native_gradient, atol=3e-5, rtol=3e-5)
    result = jax.quantum.execute()
    assert result.compatibility["selected_backend"] == "jax"


def test_energy_model_preserves_quantum_forward_hooks_and_gradients() -> None:
    model = VariationalEnergyModel()
    baseline = model()
    with model.quantum.register_forward_hook(lambda module, inputs, output: output + 1):
        actual = model()
    torch.testing.assert_close(actual, baseline + 1)
    parameters = model.quantum.parameters_tensor
    assert parameters is not None
    expected_gradient = torch.autograd.grad(baseline, parameters)[0]
    actual_gradient = torch.autograd.grad(actual, parameters)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_energy_model_rejects_non_tensor_quantum_output() -> None:
    model = VariationalEnergyModel()
    with model.quantum.register_forward_hook(
        lambda module, inputs, output: {"value": output}
    ):
        with pytest.raises(TypeError, match="quantum energy layer must return"):
            model()


def test_classifier_policy_switch_does_not_change_model_class() -> None:
    model = HybridQuantumClassifier()
    assert model.quantum.policy.mode == "statevector"
    model.set_runtime_policy(
        fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(mode="statevector"),
            observable_wires=(1,),
        )
    )
    assert isinstance(model, HybridQuantumClassifier)
    assert model.quantum.policy.mode == "statevector"


def test_hybrid_policy_switch_synchronizes_jax_cache_and_debug_hook() -> None:
    model = HybridQuantumClassifier()
    model.quantum._jax_kernel = object()
    model.quantum._jax_kernel_signature = ("stale",)
    model.set_runtime_policy(fq.RuntimePolicy(correctness_debug=True))
    assert model.quantum._jax_kernel is None
    assert model.quantum._jax_kernel_signature is None
    assert model.quantum._correctness_debug_hook_handle is not None
    model.set_runtime_policy(fq.RuntimePolicy(correctness_debug=False))
    assert model.quantum._correctness_debug_hook_handle is None
