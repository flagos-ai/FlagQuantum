from __future__ import annotations

import copy
import io

import pytest
import torch

import flagquantum as fq
import flagquantum.errors as fqe
import flagquantum.training as fqt
from flagquantum.models import HybridQuantumClassifier
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
)

pytestmark = pytest.mark.unit


def _build(parameters: torch.Tensor) -> fq.Circuit:
    return fq.Circuit(1).ry(0, parameters[0])


@pytest.mark.parametrize(
    ("error_type", "builtin_type", "category"),
    (
        (fqe.ValidationError, ValueError, "validation"),
        (fqe.PlanningError, ValueError, "planning"),
        (fqe.SerializationError, ValueError, "serialization"),
        (fqe.CompilationError, RuntimeError, "compilation"),
        (fqe.ExecutionError, RuntimeError, "execution"),
        (fqe.CapabilityError, NotImplementedError, "capability"),
    ),
)
def test_error_categories_preserve_builtin_compatibility(
    error_type: type[Exception], builtin_type: type[Exception], category: str
) -> None:
    error = error_type("failure")

    assert isinstance(error, fqe.FlagQuantumError)
    assert isinstance(error, builtin_type)
    assert error.category == category


def test_existing_domain_errors_map_to_stable_categories() -> None:
    with pytest.raises(fqe.ValidationError):
        fq.CircuitIR(n_wires=0, instructions=())
    with pytest.raises(fqe.SerializationError):
        fq.CircuitIR.from_json("not-json")

    plan = fq.plan(fq.Circuit(1))
    payload = copy.deepcopy(plan.to_dict())
    payload["version"] = "0.0"
    with pytest.raises(fqe.PlanningError) as captured:
        fq.ExecutionPlan.from_dict(payload)
    assert isinstance(captured.value, ExecutionPlanContractError)

    with pytest.raises(fqe.ExecutionError):
        fq.ExecutionResult().require_value()
    assert issubclass(fqt.TrainingStateError, fqe.ExecutionError)


def test_validation_and_capability_failures_are_stage_specific() -> None:
    with pytest.raises(fqe.ValidationError):
        fq.ExecutionOptions(batch_size=0)
    with pytest.raises(fqe.ValidationError):
        fq.RuntimePolicy(observable="unknown")  # type: ignore[arg-type]
    with pytest.raises(fqe.CapabilityError):
        fq.plan(
            fq.Circuit(1),
            options=fq.ExecutionOptions(backend="jax"),
        )
    with pytest.raises(TypeError):
        fq.plan(object())  # type: ignore[arg-type]


def test_backend_native_failure_is_preserved_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(fq.Circuit(1))

    def fail(*args, **kwargs):
        raise OSError("backend-private failure")

    monkeypatch.setattr("flagquantum.runtime.execution.run_native", fail)
    with pytest.raises(fqe.ExecutionError) as captured:
        fq.run(plan)

    assert isinstance(captured.value.__cause__, OSError)
    assert "backend-private failure" not in str(captured.value)

    def broken_builder(parameters: torch.Tensor) -> fq.Circuit:
        raise OSError("builder-private failure")

    module = fq.Module(broken_builder, 1)
    with pytest.raises(fqe.ExecutionError) as module_error:
        module.execute()
    assert isinstance(module_error.value.__cause__, OSError)


def test_module_does_not_own_deployment_binding() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        fq.Module(_build, 1, deployment_binding={})  # type: ignore[call-arg]

    module = fq.Module(_build, 1)
    module.set_extra_state(
        {
            "policy": module.policy.to_dict(),
            "deployment_binding": {"provider": "legacy"},
        }
    )
    assert not hasattr(module, "deployment_binding")
    assert set(module.get_extra_state()) == {"policy"}


def test_application_model_owns_and_checkpoints_deployment_binding() -> None:
    source = HybridQuantumClassifier(
        deployment_binding={"provider": "local", "target": "simulator"}
    )
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored = HybridQuantumClassifier(deployment_binding={"provider": "different"})

    restored.load_state_dict(torch.load(buffer, weights_only=True))

    assert restored.deployment_parameters()["binding"] == {
        "provider": "local",
        "target": "simulator",
    }
    assert not hasattr(restored.quantum, "deployment_binding")
