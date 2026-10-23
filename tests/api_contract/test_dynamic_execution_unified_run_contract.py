"""Approved contract for executing DynamicCircuit through the stable entry.

The native dynamic executor remains the numerical and performance oracle; these
tests do not promote it or expose its result type through the stable API.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "dynamic-execution-unified-run-v1-candidate.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _feedback_program() -> DynamicCircuit:
    circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
    return circuit.conditional("x", 1, classical_bit=0)


def test_contract_records_exact_approval_and_performance_stage() -> None:
    contract = _contract()

    assert contract["status"] == "approved_for_staged_implementation"
    assert contract["approval"] == {
        "approved_by": "API owner via explicit user directive",
        "approved_at": "2026-10-23",
        "directive": "approve DYNAMIC-EXECUTION-UNIFIED-RUN-CONTRACT",
        "approved_scope": (
            "staged local DynamicCircuit planning and execution through "
            "fq.plan and fq.run"
        ),
    }
    assert contract["delivery_stage"] == {
        "name": "performance_and_non_regression",
        "runtime_activation": True,
        "public_behavior_change": False,
        "successor": "remote_capability_proposal",
    }


def test_contract_records_bounded_local_performance_evidence() -> None:
    evidence = _contract()["performance_evidence"]

    assert evidence["runner"] == "flagquantum-benchmark run dynamic_unified_run"
    assert evidence["schema"] == (
        "flagquantum.dynamic_unified_run.local_performance.v1"
    )
    assert evidence["budgets"] == {
        "planning_p95_seconds": 0.05,
        "explicit_over_native_median": 3.0,
        "automatic_over_explicit_median": 2.0,
        "static_automatic_over_explicit_median": 2.0,
    }
    assert evidence["claim_exclusions"] == [
        "distributed scalability",
        "accelerator performance",
        "QPU performance",
        "release performance",
    ]
    artifact = json.loads((ROOT / evidence["artifact"]).read_text(encoding="utf-8"))
    assert artifact["schema"] == evidence["schema"]
    assert artifact["budgets"] == evidence["budgets"]
    assert artifact["budget_passed"] is True
    assert artifact["distribution_semantics"] == "single_process"
    assert artifact["scalability_claim_allowed"] is False
    assert artifact["release_gate_allowed"] is False


def test_contract_is_additive_and_reuses_stable_authorities() -> None:
    contract = _contract()
    compatibility = contract["compatibility"]

    assert compatibility == {
        "additive_behavior_only": True,
        "public_signature_change": False,
        "root_export_change": False,
        "circuit_ir_schema_change": False,
        "execution_plan_schema_change": False,
        "execution_result_schema_change": False,
        "dynamic_circuit_and_circuit_remain_distinct": True,
        "run_dynamic_remains_experimental": True,
    }
    assert contract["existing_authorities"] == {
        "program": "flagquantum.dynamic.DynamicCircuit",
        "public_ir": "flagquantum.CircuitIR@1.0",
        "options": "flagquantum.ExecutionOptions",
        "planning": "flagquantum.plan",
        "plan": "flagquantum.ExecutionPlan@1.0",
        "execution": "flagquantum.run",
        "result": "flagquantum.ExecutionResult",
        "current_oracle": "flagquantum.experimental.dynamic.run_dynamic",
    }
    assert list(inspect.signature(fq.plan).parameters) == [
        "program",
        "options",
        "outputs",
        "noise_model",
    ]
    assert "DynamicCircuit" not in fq.__all__
    assert "run_dynamic" not in fq.__all__


def test_contract_fixes_one_plan_and_result_model() -> None:
    contract = _contract()
    workflows = contract["stable_workflows"]
    planning = contract["planning"]
    result = contract["result_projection"]

    assert workflows["automatic"] == "fq.run(circuit, options=options)"
    assert workflows["explicit"] == "fq.run(fq.plan(circuit, options=options))"
    assert workflows["activation_order"] == [
        "dynamic planning",
        "local fq.run activation",
    ]
    assert planning["program_representation"] == "CircuitIR 1.0"
    assert planning["plan_representation"] == "ExecutionPlan 1.0"
    assert planning["new_public_plan_type"] is False
    assert planning["silent_static_lowering"] is False
    assert result["type"] == "flagquantum.ExecutionResult"
    assert result["measurement_kinds"] == [
        "sample",
        "mid_circuit_measurement",
    ]
    assert result["state"] is None
    assert result["value"] is None
    assert result["plan"] == "exact effective ExecutionPlan"


def test_current_dynamic_oracle_proves_the_result_projection() -> None:
    native = fq.experimental.dynamic.run_dynamic(_feedback_program(), shots=32, seed=7)
    projected = native.to_execution_result()

    assert isinstance(projected, fq.ExecutionResult)
    assert projected.samples is native.final_samples
    assert projected.samples.shape == (32, 2)
    assert tuple(item.kind for item in projected.measurements) == (
        "sample",
        "mid_circuit_measurement",
    )
    assert projected.measurements[0].shots == 32
    assert projected.measurements[1].shots == 32
    assert projected.state is None
    assert projected.value is None
    assert projected.plan is None
    assert projected.runtime["shots"] == 32
    assert projected.runtime["seed"] == 7
    assert projected.runtime["mid_circuit_measurements_available"] is True
    assert torch.equal(native.samples[:, 0], native.classical_bits[:, 0])
    assert torch.equal(native.samples[:, 1], native.classical_bits[:, 0])


def test_stable_execution_uses_one_plan_for_automatic_and_explicit_runs() -> None:
    contract = _contract()

    assert contract["delivery_stage"]["runtime_activation"] is True
    circuit = _feedback_program()
    options = fq.ExecutionOptions(shots=32, seed=7)
    plan = fq.plan(
        circuit,
        options=options,
    )
    restored = type(plan).from_json(plan.to_json())

    explicit = fq.run(restored)
    automatic = fq.run(circuit, options=options)

    assert explicit.plan is restored
    assert automatic.plan is not None
    assert explicit.plan.identity == automatic.plan.identity == plan.identity
    assert torch.equal(explicit.samples, automatic.samples)
    assert tuple(item.kind for item in explicit.measurements) == (
        "sample",
        "mid_circuit_measurement",
    )
    assert explicit.state is None
    assert explicit.value is None
    assert explicit.provenance == {}
    assert not hasattr(explicit, "final_states")
    assert not hasattr(explicit, "feedback_traces")


def test_direct_shots_keyword_reaches_dynamic_execution_only() -> None:
    result = fq.run(_feedback_program(), shots=16)

    assert isinstance(result, fq.ExecutionResult)
    assert result.samples is not None
    assert result.samples.shape == (16, 2)
    assert result.runtime["shots"] == 16

    with pytest.raises(TypeError, match="requires fq.samples"):
        fq.run(fq.Circuit(1), shots=16)


def test_direct_shots_composes_with_seed_options_without_duplication() -> None:
    circuit = _feedback_program()

    direct = fq.run(circuit, shots=32, options=fq.ExecutionOptions(seed=11))
    explicit = fq.run(
        circuit,
        options=fq.ExecutionOptions(shots=32, seed=11),
    )

    assert torch.equal(direct.samples, explicit.samples)
    assert direct.plan is not None
    assert explicit.plan is not None
    assert direct.plan.identity == explicit.plan.identity


def test_restored_plan_preserves_batch_and_precision_constraints() -> None:
    circuit = DynamicCircuit(1, bsz=2, dtype=torch.complex128)
    circuit.x(0).measure(0, classical_bit=0)
    plan = fq.plan(circuit, options=fq.ExecutionOptions(shots=8, seed=3))

    result = fq.run(type(plan).from_json(plan.to_json()))

    assert result.samples is not None
    assert result.samples.shape == (2, 8, 1)
    assert torch.all(result.samples == 1)
    assert result.plan is not None
    assert result.plan.batch_size == 2
    assert result.plan.precision == "complex128"


def test_static_planning_and_execution_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_dynamic_execution(*args: object, **kwargs: object) -> None:
        raise AssertionError("static execution entered the dynamic executor")

    monkeypatch.setattr(
        "flagquantum.runtime.dynamic.execution.run_dynamic",
        reject_dynamic_execution,
    )
    circuit = fq.Circuit(1).x(0)
    options = fq.ExecutionOptions(target="samples", shots=8, seed=3)
    plan = fq.plan(circuit, options=options)

    explicit = fq.run(plan)
    automatic = fq.run(circuit, options=options)

    assert explicit.plan is plan
    assert automatic.plan is not None
    assert automatic.plan.identity == plan.identity
    assert torch.equal(explicit.samples, automatic.samples)


def test_contract_requires_early_typed_refusal_without_partial_work() -> None:
    fail_closed = _contract()["fail_closed"]

    assert fail_closed["gradients"] == "CapabilityError before execution"
    assert fail_closed["distributed_execution"] == (
        "CapabilityError: dynamic_conditional_execution_unsupported"
    )
    assert fail_closed["unsupported_remote_target"] == "refused before submission"
    assert fail_closed["rng_consumed_on_refusal"] is False
    assert fail_closed["task_submitted_on_refusal"] is False
    assert fail_closed["partial_result_on_refusal"] is False
