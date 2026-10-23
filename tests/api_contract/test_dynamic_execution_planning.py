"""Conformance tests for the approved DynamicCircuit planning slice."""

from __future__ import annotations

import copy

import pytest
import torch

import flagquantum as fq
import flagquantum.errors as fqe
import flagquantum.noise as fqn
from flagquantum.dynamic import DynamicCircuit
from flagquantum.noise import bit_flip_channel
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    plan_program,
)

pytestmark = pytest.mark.unit


def _feedback_program() -> DynamicCircuit:
    circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
    circuit.reset(0)
    return circuit.conditional("x", 1, classical_bit=0)


def test_dynamic_plan_is_deterministic_and_uses_existing_plan_v1() -> None:
    options = fq.ExecutionOptions(shots=128, seed=7)

    first = fq.plan(_feedback_program(), options=options)
    second = fq.plan(_feedback_program(), options=options)

    assert isinstance(first, fq.ExecutionPlan)
    assert first.schema_version == "1.0"
    assert first.identity == second.identity
    assert first.mode == "statevector"
    assert first.backend == "pytorch"
    assert first.device == "cpu"
    assert first.world_size == 1
    assert first.is_distributed is False


def test_dynamic_plan_round_trip_preserves_program_options_and_identity() -> None:
    circuit = _feedback_program()
    plan = fq.plan(circuit, options=fq.ExecutionOptions(shots=64, seed=11))
    restored = type(plan).from_json(plan.to_json())
    source = plan_program(restored)
    payload = restored.to_dict()

    assert restored.identity == plan.identity
    assert source.instructions == circuit.to_ir().instructions
    assert [item.name for item in source.instructions] == [
        "h",
        "measure",
        "reset",
        "x",
    ]
    assert source.instructions[1].metadata == {
        "is_dynamic": True,
        "classical_bit": 0,
    }
    assert source.instructions[3].metadata["conditions"] == ((0, 1),)
    assert source.measurements == ()
    assert payload["resolved_options"]["shots"] == 64
    assert payload["resolved_options"]["seed"] == 11
    assert payload["decision"]["mode"] == "statevector"


def test_dynamic_plan_tampering_still_fails_under_plan_v1() -> None:
    plan = fq.plan(_feedback_program(), options=fq.ExecutionOptions(shots=16, seed=3))
    payload = copy.deepcopy(plan.to_dict())
    payload["program"]["instructions"][1]["metadata"]["classical_bit"] = 1

    with pytest.raises(ExecutionPlanContractError) as captured:
        type(plan).from_dict(payload)

    assert captured.value.reason_code == "program_fingerprint_mismatch"


def test_dynamic_plan_and_direct_run_use_the_same_stable_result_contract() -> None:
    circuit = _feedback_program()
    options = fq.ExecutionOptions(shots=16, seed=3)
    plan = fq.plan(circuit, options=options)

    explicit = fq.run(plan)
    automatic = fq.run(circuit, options=options)

    assert isinstance(explicit, fq.ExecutionResult)
    assert isinstance(automatic, fq.ExecutionResult)
    assert explicit.plan is plan
    assert automatic.plan is not None
    assert explicit.plan.identity == automatic.plan.identity
    assert torch.equal(explicit.samples, automatic.samples)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        (fq.ExecutionOptions(), "requires a positive shots"),
        (fq.ExecutionOptions(shots=8, mode="mps"), "mode='auto' only"),
        (
            fq.ExecutionOptions(shots=8, backend="jax"),
            "backend='auto' or backend='pytorch' only",
        ),
        (fq.ExecutionOptions(shots=8, device="cuda"), "local CPU device only"),
        (
            fq.ExecutionOptions(shots=8, target="state"),
            "target='auto' or target='samples' only",
        ),
        (fq.ExecutionOptions(shots=8, batch_size=2), "batch size from DynamicCircuit"),
        (
            fq.ExecutionOptions(shots=8, precision="complex128"),
            "precision from DynamicCircuit",
        ),
        (fq.ExecutionOptions(shots=8, require_gradients=True), "support gradients"),
        (fq.ExecutionOptions(shots=8, allow_approximate=True), "allow approximation"),
        (
            fq.ExecutionOptions(shots=8, allow_backend_fallback=True),
            "allow backend fallback",
        ),
    ),
)
def test_dynamic_planning_rejects_options_outside_the_local_slice(
    options: fq.ExecutionOptions, message: str
) -> None:
    with pytest.raises((fqe.CapabilityError, fqe.ValidationError), match=message):
        fq.plan(_feedback_program(), options=options)


def test_dynamic_planning_rejects_outputs_before_compilation() -> None:
    with pytest.raises(fqe.CapabilityError, match="does not accept outputs"):
        fq.plan(
            _feedback_program(),
            options=fq.ExecutionOptions(shots=8),
            outputs=fq.samples(),
        )


def test_dynamic_planning_rejects_gradient_carrying_parameters() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = DynamicCircuit(1).rx(0, theta=theta).measure(0, classical_bit=0)

    with pytest.raises(fqe.CapabilityError, match="support gradients"):
        fq.plan(circuit, options=fq.ExecutionOptions(shots=8))


def test_dynamic_planning_rejects_unserialized_custom_initial_states() -> None:
    circuit = DynamicCircuit(
        1,
        inputs=torch.tensor([[0.0, 1.0]], dtype=torch.complex64),
    ).measure(0, classical_bit=0)

    with pytest.raises(fqe.CapabilityError, match="custom initial states"):
        fq.plan(circuit, options=fq.ExecutionOptions(shots=8))


def test_dynamic_planning_validates_classical_reads_before_option_resolution() -> None:
    unread = DynamicCircuit.from_ir(
        fq.CircuitIR(
            1,
            (fq.Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),),
        )
    )

    with pytest.raises(fqe.ValidationError, match="read before measurement"):
        fq.plan(unread)


def test_dynamic_planning_keeps_noise_closed_until_execution_support_lands() -> None:
    noise_model = fqn.NoiseModel().add("x", bit_flip_channel(0.1))

    with pytest.raises(fqe.CapabilityError, match="does not yet support noise"):
        fq.plan(
            _feedback_program(),
            options=fq.ExecutionOptions(shots=8),
            noise_model=noise_model,
        )


def test_dynamic_planning_rejects_distributed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DistributedEnvironment:
        effective_world_size = 2

    monkeypatch.setattr(
        "flagquantum.runtime.distributed.backend_policy."
        "resolve_distributed_backend_policy",
        lambda: DistributedEnvironment(),
    )

    with pytest.raises(
        fqe.CapabilityError,
        match="dynamic_conditional_execution_unsupported.*world_size=1",
    ):
        fq.plan(_feedback_program(), options=fq.ExecutionOptions(shots=8))
