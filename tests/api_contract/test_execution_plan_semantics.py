from __future__ import annotations

import copy

import pytest
import torch

import flagquantum as fq
import flagquantum.noise as fqn
from flagquantum.compilation.execution_plan_contract import (
    ExecutionPlanContractError,
)

pytestmark = pytest.mark.unit


def _bell() -> fq.Circuit:
    return fq.Circuit(2).h(0).cx(0, 1)


def test_plan_identity_is_deterministic_and_exposes_final_decision() -> None:
    options = fq.ExecutionOptions(mode="statevector", precision="complex64")

    first = fq.plan(_bell(), options=options)
    second = fq.plan(_bell(), options=options)

    assert first.identity == second.identity
    assert len(first.identity) == 64
    assert first.schema_version == "1.0"
    assert first.mode == "statevector"
    assert first.backend == "pytorch"
    assert first.target == "auto"
    assert first.batch_size == 1
    assert first.precision == "complex64"
    assert first.is_distributed is False


def test_plan_json_round_trip_preserves_identity_and_execution() -> None:
    plan = fq.plan(_bell(), options=fq.ExecutionOptions(mode="statevector"))

    restored = type(plan).from_json(plan.to_json())
    result = fq.run(restored)

    assert restored.identity == plan.identity
    assert result.plan is restored
    assert torch.allclose(
        result.state,
        torch.tensor([[2**-0.5, 0, 0, 2**-0.5]], dtype=torch.complex64),
    )


def test_plan_payload_tampering_fails_closed() -> None:
    plan = fq.plan(_bell())
    payload = copy.deepcopy(plan.to_dict())
    payload["decision"]["precision"] = "complex128"

    with pytest.raises(ExecutionPlanContractError) as captured:
        type(plan).from_dict(payload)

    assert captured.value.reason_code in {
        "environment_incompatible",
        "identity_mismatch",
    }


def test_plan_payload_unknown_field_and_version_fail_closed() -> None:
    plan = fq.plan(_bell())
    unknown = copy.deepcopy(plan.to_dict())
    unknown["unexpected"] = True
    old_version = copy.deepcopy(plan.to_dict())
    old_version["version"] = "0.9"

    with pytest.raises(ExecutionPlanContractError) as unknown_error:
        type(plan).from_dict(unknown)
    with pytest.raises(ExecutionPlanContractError) as version_error:
        type(plan).from_dict(old_version)

    assert unknown_error.value.reason_code == "unsupported_schema"
    assert version_error.value.reason_code == "unsupported_schema"


def test_run_plan_does_not_replan_or_recompile(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = fq.plan(_bell(), options=fq.ExecutionOptions(mode="statevector"))

    def forbidden(*args, **kwargs):
        raise AssertionError("planner/compiler must not run for fq.run(plan)")

    monkeypatch.setattr("flagquantum.runtime.planner.plan", forbidden)
    monkeypatch.setattr("flagquantum.runtime.execution.compile_for_backend", forbidden)
    monkeypatch.setattr(
        "flagquantum.runtime.execution.select_execution_mode", forbidden
    )
    monkeypatch.setattr("flagquantum.runtime.execution.build_plan", forbidden)

    result = fq.run(plan)

    assert result.plan is plan


def test_run_noisy_plan_uses_planned_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noise_model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.25))
    plan = fq.plan(
        fq.Circuit(1).x(0),
        options=fq.ExecutionOptions(mode="density_matrix"),
        noise_model=noise_model,
    )
    restored = type(plan).from_json(plan.to_json())

    def forbidden(*args, **kwargs):
        raise AssertionError("noise lowering must not run for fq.run(plan)")

    monkeypatch.setattr("flagquantum.runtime.execution.lower_noise_model", forbidden)

    result = fq.run(restored)

    assert result.plan is restored
    assert result.state is not None


def test_plan_rejects_changed_world_size_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(_bell())

    class ChangedEnvironment:
        effective_world_size = 2

    monkeypatch.setattr(
        "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
        lambda: ChangedEnvironment(),
    )

    with pytest.raises(ExecutionPlanContractError) as captured:
        fq.run(plan)

    assert captured.value.reason_code == "world_size_mismatch"


def test_automatic_and_explicit_plan_paths_are_equivalent() -> None:
    options = fq.ExecutionOptions(mode="statevector")
    circuit = _bell()

    explicit_plan = fq.plan(circuit, options=options)
    explicit = fq.run(explicit_plan)
    automatic = fq.run(circuit, options=options)

    assert explicit.plan.identity == explicit_plan.identity
    assert automatic.plan.identity == explicit_plan.identity
    assert torch.equal(automatic.state, explicit.state)
    assert automatic.provenance == explicit.provenance


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("options", fq.ExecutionOptions()),
        ("measurements", ()),
        ("noise_model", object()),
    ],
)
def test_plan_input_rejects_all_semantic_overrides(keyword: str, value: object) -> None:
    plan = fq.plan(_bell())

    with pytest.raises(TypeError, match=f"{keyword} must be None"):
        fq.run(plan, **{keyword: value})


def test_plan_with_sampling_request_executes_without_runtime_override() -> None:
    plan = fq.plan(
        _bell(),
        options=fq.ExecutionOptions(target="samples", shots=16, seed=7),
    )

    first = fq.run(plan)
    second = fq.run(type(plan).from_dict(plan.to_dict()))

    assert first.samples is not None
    assert first.samples.shape == (1, 16, 2)
    assert torch.equal(first.samples, second.samples)
