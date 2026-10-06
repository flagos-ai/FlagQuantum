from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from examples.qdiffusion_kaiwu import validate_acceptance as validator_module
from examples.qdiffusion_kaiwu.validate_acceptance import (
    COMPONENT_FIELDS_BY_SCHEMA,
    CONFIG_FIELDS,
    CONFIG_SECTION_FIELDS,
    FINAL_ACCEPTANCE_LIMITATIONS,
    METRIC_NAMES,
    _validate_component_bundle,
    _validate_component_field_set,
    _validate_config,
    _validate_executable_component_nested_fields,
    _validate_execution_component_time,
    _validate_final_record_field_sets,
    _validate_precision_evidence,
    _validate_provider_resource_gate_binding,
    _validate_provider_result_schema,
    _validate_provider_smoke_component,
    _validate_receipt_submission_floor,
    _validate_sampling_receipt,
    _validate_system_record,
    validate_acceptance,
)

pytestmark = pytest.mark.unit

_REVISION = "a" * 40
_PLUGIN_REVISION = "b" * 40


@pytest.mark.parametrize("schema", tuple(COMPONENT_FIELDS_BY_SCHEMA))
def test_executable_component_field_sets_are_closed(schema: str) -> None:
    expected = COMPONENT_FIELDS_BY_SCHEMA[schema]
    complete = dict.fromkeys(expected)
    complete["schema"] = schema
    errors: list[str] = []

    _validate_component_field_set(complete, label="component", errors=errors)

    assert errors == []
    for mutation in (
        {key: value for key, value in complete.items() if key != "recorded_at"},
        {**complete, "unexpected_secret_field": "must-not-pass"},
    ):
        errors = []
        _validate_component_field_set(mutation, label="component", errors=errors)
        assert errors == ["component: field set differs from its closed schema"]


def test_provider_resource_gate_binding_recomputes_the_declared_budget() -> None:
    snapshot = {
        "schema": "flagquantum.qboson_provider_resources",
        "version": "1.0",
        "source": "authenticated_resource_bill",
        "captured_at": "2026-10-05T00:00:00+00:00",
        "valid_until": "2026-10-06T00:00:00+00:00",
        "resources": [
            {
                "target": target,
                "mode": mode,
                "available": 127 if mode == "sampling" else 1,
                "used": 0,
            }
            for target in ("SPQC-1", "SPQC-550", "SPQC-1000")
            for mode in ("optimization", "sampling")
        ],
        "claim_boundary": (
            "Account-resource observation only; it is not spend approval, project "
            "assignment, provider evidence, execution evidence, or acceptance evidence."
        ),
    }
    record = {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "remote_call_budget": 128,
        "provider_resource_gate": {
            "snapshot_sha256": "a" * 64,
            "checked_at": "2026-10-05T12:00:00+00:00",
            "mode": "sampling",
            "required_calls": 128,
        },
        "task_receipts": [
            {"submitted_at": "2026-10-05T13:00:00+00:00"}
        ],
    }
    errors: list[str] = []

    _validate_provider_resource_gate_binding(
        record,
        label="system",
        provider_resources={"a" * 64: snapshot},
        recorded_at=datetime(2026, 10, 5, 14, tzinfo=timezone.utc),
        errors=errors,
    )

    assert errors == [
        "system: provider resource gate failed: "
        "sampling_resource_budget_insufficient"
    ]

    for resource in snapshot["resources"]:
        if resource["mode"] == "sampling":
            resource["available"] = 128
    record["task_receipts"] = [
        {"submitted_at": "2026-10-06T01:00:00+00:00"}
    ]
    errors = []
    _validate_provider_resource_gate_binding(
        record,
        label="system",
        provider_resources={"a" * 64: snapshot},
        recorded_at=datetime(2026, 10, 6, 2, tzinfo=timezone.utc),
        errors=errors,
    )

    assert errors == [
        "system: remote receipt 0 follows resource snapshot expiry"
    ]

    record["provider_resource_gate"]["checked_at"] = (
        "2026-10-04T23:59:00+00:00"
    )
    record["task_receipts"] = [
        {"submitted_at": "2026-10-04T23:59:30+00:00"}
    ]
    errors = []
    _validate_provider_resource_gate_binding(
        record,
        label="system",
        provider_resources={"a" * 64: snapshot},
        recorded_at=datetime(2026, 10, 5, 1, tzinfo=timezone.utc),
        errors=errors,
    )

    assert (
        "system: remote receipt 0 predates resource snapshot capture" in errors
    )


def test_provider_smoke_tasks_cannot_predate_the_resource_snapshot() -> None:
    config = _config()
    snapshot = {
        "schema": "flagquantum.qboson_provider_resources",
        "version": "1.0",
        "source": "authenticated_resource_bill",
        "captured_at": "2026-10-05T12:00:00+00:00",
        "valid_until": "2026-10-06T12:00:00+00:00",
        "resources": [
            {
                "target": target,
                "mode": mode,
                "available": 1,
                "used": 0,
            }
            for target in ("SPQC-1", "SPQC-550", "SPQC-1000")
            for mode in ("optimization", "sampling")
        ],
        "claim_boundary": (
            "Account-resource observation only; it is not spend approval, project "
            "assignment, provider evidence, execution evidence, or acceptance evidence."
        ),
    }
    task_template = {
        "receipt_schema": "flagquantum.kaiwu-task.v1",
        "matrix_sha256": (
            "0352923b6964d8a65fc742c5a5b251ab967d43e8c4db9e3ee3a0f2f2fa5b0487"
        ),
        "matrix_size": 2,
        "requested_samples": config["requested_samples"],
        "project_no": "CPQC-test",
        "submitted_at": "2026-10-05T11:59:59+00:00",
        "returned_samples": config["requested_samples"],
        "samples": [[1, -1] for _ in range(config["requested_samples"])],
        "energies": [2.0 for _ in range(config["requested_samples"])],
        "provider_target": "SPQC-provider",
        "raw_status": "completed",
        "fallback_occurred": False,
        "minimum_energy": 2.0,
        "maximum_energy": 2.0,
        "provider_task_id_available": True,
        "provider_target_available": True,
        "provider_result_schema": {
            "available": False,
            "reason": "test_fixture",
        },
    }
    tasks = []
    for mode in ("optimization", "sampling"):
        task = dict(task_template)
        task.update(
            {
                "task_name": f"smoke-{mode}",
                "task_mode": mode,
                "provider_task_id": f"task-{mode}",
            }
        )
        tasks.append(task)
    record = {
        "schema": "flagquantum.qboson_kaiwu_live_smoke",
        "version": "1.0",
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "transport": "kaiwu_cim",
        "real_provider_evidence": True,
        "qboson_hardware_used": True,
        "qboson_target": "SPQC-provider",
        "project_no": "CPQC-test",
        "environment_lock_sha256": config["software"]["environment_lock_sha256"],
        "sdk_approval_sha256": "d" * 64,
        "provider_resources_sha256": "c" * 64,
        "tasks": tasks,
        "run_completed": True,
        "failure": None,
        "live_provider_smoke_passed": True,
        "provider_identity_complete": True,
        "hardware_acceptance": True,
        "fallback_occurred": False,
        "secrets_redacted": True,
        "limitations": [
            "This smoke test does not execute QDiffusion or A800 tensor work.",
            "Hardware acceptance remains false without provider-reported task and target identities.",
            "This record does not establish performance, quantum advantage, or production maturity.",
        ],
    }
    errors: list[str] = []

    _validate_provider_smoke_component(
        record,
        config=config,
        sdk_approval_sha256="d" * 64,
        provider_resources=(snapshot, "c" * 64),
        errors=errors,
    )

    assert "provider smoke: task 0 submission predates its resource snapshot" in errors
    assert "provider smoke: task 1 submission predates its resource snapshot" in errors
    assert "provider smoke: task 0 submission predates the SDK rights review" in errors
    assert "provider smoke: task 1 submission predates the SDK rights review" in errors

    for task in record["tasks"]:
        task["submitted_at"] = "2026-10-06T00:00:00+00:00"
    record["tasks"][1]["task_name"] = record["tasks"][0]["task_name"]
    errors = []
    _validate_provider_smoke_component(
        record,
        config=config,
        sdk_approval_sha256="d" * 64,
        provider_resources=(snapshot, "c" * 64),
        errors=errors,
    )

    assert "provider smoke: SDK task names are not unique" in errors


@pytest.mark.parametrize(
    ("filename", "schema", "postflight_fields"),
    (
        (
            "qdiffusion_system_live.py",
            "flagquantum.qboson_qdiffusion_system_live_probe",
            frozenset(),
        ),
        (
            "qdiffusion_protein_training_live.py",
            "flagquantum.qboson_qdiffusion_protein_training",
            frozenset({"artifact_inputs_unchanged"}),
        ),
        (
            "qdiffusion_protein_evaluate.py",
            "flagquantum.qboson_qdiffusion_protein_evaluation",
            frozenset(),
        ),
        (
            "qdiffusion_portability_replay_live.py",
            "flagquantum.qboson_qdiffusion_portability_replay",
            frozenset({"artifact_inputs_unchanged"}),
        ),
    ),
)
def test_closed_component_fields_match_producer_payloads(
    filename: str, schema: str, postflight_fields: frozenset[str]
) -> None:
    source = Path(__file__).parents[3] / "examples" / "qdiffusion_kaiwu" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    candidates: list[set[str]] = []
    assigned_payload_fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            targets = node.targets if isinstance(node, ast.Assign) else ()
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "payload"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    assigned_payload_fields.add(target.slice.value)
            continue
        fields = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "schema" in fields:
            candidates.append(fields)

    producer_fields = (
        max(candidates, key=len) | postflight_fields | assigned_payload_fields
    )

    assert producer_fields == COMPONENT_FIELDS_BY_SCHEMA[schema]


def test_provider_result_schema_rejects_values_and_inconsistent_descriptions() -> None:
    valid = {
        "available": True,
        "result": {
            "type": "builtins.dict",
            "length": 1,
            "string_keys": True,
            "fields_safe": True,
            "fields": ["solutions"],
            "field_schemas": {
                "solutions": {
                    "type": "builtins.list",
                    "length": 2,
                    "element_types": ["builtins.list"],
                    "element_type_scan_limit": 64,
                }
            },
        },
    }
    errors: list[str] = []
    _validate_provider_result_schema(valid, label="schema", errors=errors)
    assert errors == []

    for mutation in (
        {**valid, "raw_result": {"sdk_code": "secret"}},
        {
            **valid,
            "result": {
                **valid["result"],
                "field_schemas": {
                    **valid["result"]["field_schemas"],
                    "undeclared": {"type": "builtins.str", "length": 6},
                },
            },
        },
        {"available": False, "reason": "credential\nvalue"},
    ):
        errors = []
        _validate_provider_result_schema(mutation, label="schema", errors=errors)
        assert errors


def test_final_host_record_and_nested_field_sets_are_closed() -> None:
    config_hash = "c" * 64
    environment_hash = "d" * 64
    records = (
        _record("jp-a800-171", "primary", config_hash, environment_hash),
        _record(
            "jp-a800-172", "portability_replay", config_hash, environment_hash
        ),
    )
    for record in records:
        errors: list[str] = []
        _validate_final_record_field_sets(record, label="record", errors=errors)
        assert errors == []

        for path in (
            (),
            ("artifacts",),
            ("precision_policy",),
            ("training",),
            ("generation",),
            ("transfer_accounting",),
            ("transfer_accounting", "sampler_boundaries", 0),
            ("acceptance",),
            (
                "application_evidence",
                "records",
                0,
            )
            if record["run_role"] == "primary"
            else ("portability_evidence",),
        ):
            changed = copy.deepcopy(record)
            target: Any = changed
            for part in path:
                target = target[part]
            target["unexpected_secret_field"] = "must-not-pass"
            errors = []
            _validate_final_record_field_sets(
                changed, label="record", errors=errors
            )
            assert any("closed schema" in error for error in errors)


def test_final_host_record_rejects_changed_claim_limitations() -> None:
    record = _record("jp-a800-171", "primary", "c" * 64, "d" * 64)
    record["limitations"].append("production-ready")
    config = _config()
    config["software"]["environment_lock_sha256"] = "d" * 64
    errors: list[str] = []

    _validate_system_record(
        record,
        config=config,
        config_sha256="c" * 64,
        label="record",
        errors=errors,
    )

    assert "record: final claim limitations differ" in errors


def test_system_component_nested_fields_reject_extensions_and_false_failures() -> None:
    record = {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 1,
            "scale_factor_min": 1.0,
            "scale_factor_max": 1.0,
            "max_abs_error": 0.0,
            "mean_of_matrix_mean_abs_error": 0.0,
        },
        "failure": None,
        "provider_result_schema": {"available": False, "reason": "test_fixture"},
        "training": {
            "objective": -0.5,
            "gradient_norm": 1.0,
            "parameter_delta_max": 0.1,
        },
        "generation": {
            "generated_tokens": [[1, 2, 3]],
            "token_constraints_passed": True,
        },
        "transfer_accounting": {
            "matrix_origin_device": "cuda:0",
            "sampler_boundaries": [],
            "returned_sample_target_device": "cuda:0",
        },
        "limitations": [
            "This bounded system probe does not run the frozen protein effectiveness experiment.",
            "System acceptance remains failed without provider-reported task and target identities.",
            "Two independent passing host records are required; this is one single-device run.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }
    errors: list[str] = []
    _validate_executable_component_nested_fields(
        record, label="system", errors=errors
    )
    assert errors == []

    mutations = []
    for field in (
        "precision_policy",
        "training",
        "generation",
        "transfer_accounting",
    ):
        changed = copy.deepcopy(record)
        changed[field]["unexpected_secret_field"] = "must-not-pass"
        mutations.append(changed)
    changed = copy.deepcopy(record)
    changed["provider_result_schema"]["raw_result"] = "must-not-pass"
    mutations.append(changed)
    changed = copy.deepcopy(record)
    changed["failure"] = {"type": "RuntimeError", "message": "hidden"}
    mutations.append(changed)
    changed = copy.deepcopy(record)
    changed["limitations"].append("unreviewed claim")
    mutations.append(changed)

    for mutation in mutations:
        errors = []
        _validate_executable_component_nested_fields(
            mutation, label="system", errors=errors
        )
        assert errors


def test_other_component_nested_fields_reject_extensions() -> None:
    precision_policy = {
        "name": "explicit-int8",
        "target_min": -127,
        "target_max": 127,
        "matrix_count": 1,
        "scale_factor_min": 1.0,
        "scale_factor_max": 1.0,
        "max_abs_error": 0.0,
        "mean_of_matrix_mean_abs_error": 0.0,
    }
    artifact_names = (
        "test_fasta",
        "baseline_fasta",
        "guided_fasta",
        "training_history",
        "sequence_metrics",
        "baseline_quality",
        "guided_quality",
    )
    training = {
        "schema": "flagquantum.qboson_qdiffusion_protein_training",
        "precision_policy": precision_policy,
        "failure": None,
        "workflow_artifacts": {
            name: {"relative_path": f"{name}.json", "sha256": "a" * 64}
            for name in artifact_names
        },
        "limitations": [
            "This record covers one protein-training seed only.",
            "ESM2 evaluation and two-host system acceptance are separate gates.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }
    evaluation = {
        "schema": "flagquantum.qboson_qdiffusion_protein_evaluation",
        "baseline_metrics": dict.fromkeys(METRIC_NAMES, 0.0),
        "guided_metrics": dict.fromkeys(METRIC_NAMES, 0.0),
    }
    portability = {
        "schema": "flagquantum.qboson_qdiffusion_portability_replay",
        "precision_policy": precision_policy,
        "failure": None,
        "artifacts": {
            "dataset_sha256": "a" * 64,
            "base_checkpoint_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "evaluation_model_sha256": "d" * 64,
            "trained_energy_checkpoint_sha256": "e" * 64,
        },
        "fixture": {
            "training_seed": 1701,
            "index": 0,
            "steps": 1,
            "energy_objective": -0.5,
            "generated_length": 8,
            "generated_sha256": "f" * 64,
            "token_constraints_passed": True,
        },
        "limitations": [
            "This is one fixed replay fixture, not a second training run.",
            "Final acceptance also requires both system gates and all primary-host seeds.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }
    for record in (training, evaluation, portability):
        errors: list[str] = []
        _validate_executable_component_nested_fields(
            record, label="component", errors=errors
        )
        assert errors == []

    mutations = (
        (
            training,
            lambda record: record["workflow_artifacts"]["test_fasta"].update(
                unexpected_secret_field="must-not-pass"
            ),
        ),
        (
            evaluation,
            lambda record: record["guided_metrics"].update(
                unexpected_secret_field=0.0
            ),
        ),
        (
            portability,
            lambda record: record["fixture"].update(
                unexpected_secret_field="must-not-pass"
            ),
        ),
    )
    for source, mutate in mutations:
        changed = copy.deepcopy(source)
        mutate(changed)
        errors = []
        _validate_executable_component_nested_fields(
            changed, label="component", errors=errors
        )
        assert errors


def test_execution_component_time_is_bound_to_prerequisites_and_receipts() -> None:
    config = _config()
    valid = {
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "task_receipts": [
            {"submitted_at": "2026-10-06T00:00:00+00:00"},
        ],
    }
    smoke_time = datetime.fromisoformat("2026-10-06T00:00:00+00:00")
    errors: list[str] = []

    observed = _validate_execution_component_time(
        valid,
        label="component",
        config=config,
        provider_smoke_time=smoke_time,
        errors=errors,
    )

    assert observed == smoke_time
    assert errors == []

    mutations = (
        ({**valid, "recorded_at": "2026-10-06T08:00:00+08:00"}, "aware UTC"),
        ({**valid, "recorded_at": "2026-10-05T23:59:59+00:00"}, "predates"),
        (
            {
                **valid,
                "task_receipts": [
                    {"submitted_at": "2026-10-07T00:00:00+00:00"}
                ],
            },
            "submission follows its record",
        ),
        (
            {
                **valid,
                "task_receipts": [
                    {"submitted_at": "2026-10-05T23:59:59+00:00"}
                ],
            },
            "remote receipt 0 predates the provider smoke",
        ),
    )
    for record, message in mutations:
        errors = []
        _validate_execution_component_time(
            record,
            label="component",
            config=config,
            provider_smoke_time=smoke_time,
            errors=errors,
        )
        assert any(message in error for error in errors)


def test_receipt_submission_floor_binds_cross_component_dependencies() -> None:
    record = {
        "task_receipts": [
            {"submitted_at": "2026-10-06T00:00:00+00:00"},
            {"submitted_at": "2026-10-06T00:00:02+00:00"},
        ]
    }
    errors: list[str] = []

    _validate_receipt_submission_floor(
        record,
        earliest=datetime(2026, 10, 6, 0, 0, 1, tzinfo=timezone.utc),
        label="seed 1701",
        prerequisite_label="primary system evidence",
        errors=errors,
    )

    assert errors == [
        "seed 1701: remote receipt 0 predates the primary system evidence"
    ]


def _complete_task_receipt() -> dict[str, Any]:
    return {
        "schema": "flagquantum.kaiwu-task.v1",
        "task_name": "system-task",
        "matrix_sha256": "7" * 64,
        "matrix_size": 3,
        "mode": "sampling",
        "requested_samples": 10,
        "project_no": "CPQC-test",
        "submitted_at": "2026-10-05T00:00:00+00:00",
        "provider_task_id": "provider-task",
        "provider_target": "SPQC-provider",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "wrong", "schema is unsupported"),
        ("task_name", "", "no task name"),
        ("task_name", "system\ntask", "no task name"),
        ("task_name", " system-task", "no task name"),
        ("matrix_sha256", "bad", "no matrix digest"),
        ("matrix_size", True, "invalid matrix size"),
        ("mode", "optimization", "not sampling"),
        ("requested_samples", 11, "sample count differs"),
        ("project_no", " ", "no project number"),
        ("project_no", "CPQC\ttest", "no project number"),
        ("submitted_at", "2026-10-05T08:00:00+08:00", "aware UTC"),
        ("provider_task_id", "", "no provider_task_id"),
        ("provider_task_id", "provider\ntask", "no provider_task_id"),
        ("provider_target", "", "no provider_target"),
        ("provider_target", "SPQC\u200bprovider", "no provider_target"),
    ),
)
def test_complete_sampling_receipt_rejects_tampered_identity_fields(
    field: str, value: object, message: str
) -> None:
    receipt = _complete_task_receipt()
    receipt[field] = value
    errors: list[str] = []

    _validate_sampling_receipt(
        receipt,
        label="receipt",
        expected_requested_samples=10,
        errors=errors,
    )

    assert any(message in error for error in errors)


def test_complete_sampling_receipt_rejects_missing_or_extra_fields() -> None:
    for receipt in (
        {
            key: value
            for key, value in _complete_task_receipt().items()
            if key != "schema"
        },
        {**_complete_task_receipt(), "unexpected": True},
    ):
        errors: list[str] = []
        _validate_sampling_receipt(
            receipt,
            label="receipt",
            expected_requested_samples=10,
            errors=errors,
        )
        assert any("field set is incomplete" in error for error in errors)


def _complete_precision_record() -> dict[str, Any]:
    return {
        "precision_policy": {
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 1,
            "scale_factor_min": 2.0,
            "scale_factor_max": 2.0,
            "max_abs_error": 0.25,
            "mean_of_matrix_mean_abs_error": 0.125,
        },
        "precision_evidence": [
            {
                "original_matrix_sha256": "6" * 64,
                "submission_matrix_sha256": "7" * 64,
                "source_type": "numpy.ndarray",
                "source_dtype": "float32",
                "normalized_dtype": "torch.float64",
                "normalized_min": -63.5,
                "normalized_max": 63.5,
                "symmetry_normalization": "arithmetic_mean",
                "rounding_policy": "round_half_to_even",
                "scale_factor": 2.0,
                "target_min": -127,
                "target_max": 127,
                "max_abs_error": 0.25,
                "mean_abs_error": 0.125,
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: record["precision_evidence"].clear(), "evidence is missing"),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "submission_matrix_sha256", "8" * 64
            ),
            "submissions differ from task receipts",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "rounding_policy", "truncate"
            ),
            "rounding policy is unsupported",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "source_type", "torch.Tensor"
            ),
            "source type is not numpy.ndarray",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "normalized_min", 64.0
            ),
            "normalized coefficient range is invalid",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "normalized_max", 100.0
            ),
            "scale factor differs from coefficient range",
        ),
        (
            lambda record: record["precision_policy"].__setitem__("max_abs_error", 0.5),
            "aggregate max_abs_error differs",
        ),
    ),
)
def test_per_matrix_precision_evidence_fails_closed(mutation, message: str) -> None:
    record = _complete_precision_record()
    mutation(record)
    errors: list[str] = []

    _validate_precision_evidence(
        record,
        label="precision",
        receipt_matrix_digests=["7" * 64],
        errors=errors,
    )

    assert any(message in error for error in errors)


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_config",
        "version": "1.0",
        "preregistered_at": "2026-10-05T00:00:00Z",
        "primary_host": "jp-a800-171",
        "replay_host": "jp-a800-172",
        "host_identities": {
            "jp-a800-171": "node-a800-171",
            "jp-a800-172": "node-a800-172",
        },
        "software": {
            "source_revision": _REVISION,
            "flagquantum_version": "0.2.0",
            "kaiwu_pytorch_plugin_revision": _PLUGIN_REVISION,
            "python_version": "3.10.18",
            "torch_version": "2.7.0",
            "kaiwu_sdk_version": "1.3.1",
            "environment_lock_sha256": "6" * 64,
        },
        "kaiwu_sdk": {
            "schema": "flagquantum.qboson_kaiwu_sdk_approval",
            "version": "1.1",
            "distribution": "kaiwu",
            "sdk_version": "1.3.1",
            "wheel_filename": "kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl",
            "source_url": "https://pypi.org/pypi/kaiwu/1.3.1/json",
            "sha256": "7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455",
            "service_terms_url": (
                "https://platform.qboson.com/agreement?"
                "type=QBoson-SPQC-Platform-Users-Agreement"
            ),
            "service_terms_effective_date": "2026-07-09",
            "rights_reviewed_at": "2026-10-06T00:00:00Z",
            "approval_reference": "LEGAL-APPROVAL-1",
            "project_no": "CPQC-test",
            "project_assignment_reviewed_at": "2026-10-06T00:00:00Z",
            "project_assignment_reference": "QBOSON-ASSIGNMENT-1",
            "organizational_use_approved": True,
            "isolated_container_use_approved": True,
            "host_staging_approved": True,
            "adapter_distribution_approved": True,
            "sdk_redistribution_policy": "no-sdk-redistribution",
        },
        "dataset": {
            "name": "frozen",
            "revision": "v1",
            "source_url": "https://example.test/dataset.fasta",
            "license_id": "CC-BY-4.0",
            "license_evidence_url": "https://example.test/dataset-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "split": "deterministic-shuffle-v1",
            "sha256": "c" * 64,
            "min_length": 50,
            "max_length": 256,
            "max_records": 640,
            "validation_ratio": 0.05,
            "test_ratio": 0.05,
        },
        "checkpoint": {
            "name": "dplm",
            "revision": "v1",
            "source_url": "https://example.test/dplm",
            "license_id": "Apache-2.0",
            "license_evidence_url": "https://example.test/dplm-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "sha256": "d" * 64,
        },
        "tokenizer": {
            "name": "dplm",
            "revision": "v1",
            "source_url": "https://example.test/dplm",
            "license_id": "Apache-2.0",
            "license_evidence_url": "https://example.test/dplm-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "sha256": "e" * 64,
        },
        "evaluation_model": {
            "name": "esm2_t33_650M_UR50D",
            "revision": "v1",
            "source_url": "https://example.test/esm2.pt",
            "license_id": "MIT",
            "license_evidence_url": "https://example.test/esm2-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "sha256": "f" * 64,
        },
        "training": {
            "freeze_proposal": True,
            "epochs": 20,
            "min_epochs": 3,
            "batch_size": 4,
            "num_candidates": 4,
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
            "validation_steps": 3,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "early_stop_patience": 4,
            "remote_call_budget_per_seed": 71269,
        },
        "generation": {
            "sequence_count": 32,
            "max_steps": 64,
            "num_candidates": 4,
            "proposal_temperature": 0.3,
            "proposal_noise_scale": 1.0,
            "energy_temperature": 1.25,
            "disable_resample": False,
            "resample_ratio": 0.2,
            "resample_top_p": 0.9,
            "portability_training_seed": 1701,
            "portability_fixture_index": 0,
            "portability_steps": 3,
        },
        "evaluation": {"pair_mode": "order", "pooling": "mean", "batch_size": 1},
        "seeds": [1701, 1702, 1703],
        "requested_samples": 10,
        "remote_call_budget": 128,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
        "primary_metric": {"name": "mean_cosine_distance", "direction": "lower"},
        "thresholds": {
            "uniqueness_baseline_fraction_min": 0.95,
            "repeat_ratio_absolute_increase_max": 0.05,
            "invalid_sequence_count_max": 0,
        },
    }


def _metrics(*, cosine: float, uniqueness: float, repeat: float) -> dict[str, float]:
    return {
        "mean_cosine_distance": cosine,
        "median_cosine_distance": cosine,
        "mean_l2_distance": 1.0,
        "median_l2_distance": 1.0,
        "identity_to_reference_mean": 0.4,
        "amino_acid_jsd": 0.1,
        "kmer2_jsd": 0.2,
        "kmer3_jsd": 0.3,
        "uniqueness_ratio": uniqueness,
        "repeat_ratio_ge4": repeat,
        "length_match_ratio": 1.0,
        "invalid_sequence_count": 0.0,
    }


def _record(
    host: str, role: str, config_sha256: str, environment_lock_sha256: str
) -> dict[str, Any]:
    transfer_boundary = {
        "input_type": "numpy.ndarray",
        "input_device": "cpu",
        "input_dtype": "float32",
        "matrix_shape": [3, 3],
        "original_matrix_sha256": "6" * 64,
        "submission_matrix_sha256": "7" * 64,
        "canonical_device": "cpu",
        "canonical_dtype": "torch.float64",
        "submission_storage": "cpu_python_tuple",
        "returned_storage": "cpu_numpy",
        "returned_dtype": "int8",
        "returned_shape": [10, 3],
        "cache_hit": False,
    }
    record = {
        "schema": "flagquantum.qboson_qdiffusion_acceptance",
        "version": "1.0",
        "source_revision": _REVISION,
        "flagquantum_version": "0.2.0",
        "kaiwu_pytorch_plugin_revision": _PLUGIN_REVISION,
        "source_preflight_sha256": ("8" * 64 if host == "jp-a800-171" else "9" * 64),
        "transfer_manifest_sha256": "7" * 64,
        "environment_lock_sha256": environment_lock_sha256,
        "provider_resource_gate": {
            "snapshot_sha256": "c" * 64,
            "checked_at": "2026-10-05T12:00:00+00:00",
            "mode": "sampling",
            "required_calls": 128,
        },
        "python_version": "3.10.18",
        "torch_version": "2.7.0",
        "kaiwu_sdk_version": "1.3.1",
        "experiment_config_sha256": config_sha256,
        "execution_host": host,
        "run_role": role,
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "transport": "kaiwu_cim",
        "qboson_hardware_used": True,
        "real_provider_evidence": True,
        "provider_reported_target": True,
        "qboson_target": "SPQC-provider-label",
        "qboson_task_ids": [f"task-{host}"],
        "sampling_mode": "sampling",
        "requested_samples": 10,
        "returned_samples": 10,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 2,
            "scale_factor_min": 1.0,
            "scale_factor_max": 2.0,
            "max_abs_error": 0.0,
            "mean_of_matrix_mean_abs_error": 0.0,
        },
        "remote_call_budget": 128,
        "remote_call_count": 2,
        "fallback_occurred": False,
        "retrieval_resubmitted": False,
        "secrets_redacted": True,
        "artifacts": {
            "dataset_sha256": "c" * 64,
            "base_checkpoint_sha256": "d" * 64,
            "tokenizer_sha256": "e" * 64,
            "evaluation_model_sha256": "f" * 64,
            "trained_energy_checkpoint_sha256": "1" * 64,
        },
        "training": {
            "energy_objective": -0.5,
            "gradient_norm": 1.0,
            "parameter_delta_max": 0.01,
        },
        "generation": {"token_constraints_passed": True, "invalid_sequence_count": 0},
        "transfer_accounting": {
            "matrix_origin_device": "cuda:0",
            "sampler_boundaries": [transfer_boundary, dict(transfer_boundary)],
            "returned_sample_target_device": "cuda:0",
        },
        "acceptance": {
            "system": "pass",
            "application": "pass" if role == "primary" else "not_run",
        },
        "limitations": list(FINAL_ACCEPTANCE_LIMITATIONS),
        "component_bundle_required": True,
        "system_evidence_sha256": ("a" * 64 if role == "primary" else "b" * 64),
    }
    if role == "primary":
        record["attempted_seeds"] = [1701, 1702, 1703]
        record["baseline_metrics"] = _metrics(
            cosine=0.5, uniqueness=1.0, repeat=0.0
        )
        record["guided_metrics"] = _metrics(
            cosine=0.4, uniqueness=0.96, repeat=0.04
        )
        record["application_evidence"] = {
            "aggregation": "arithmetic_mean_across_frozen_seeds",
            "records": [
                {
                    "seed": seed,
                    "training_record_sha256": str(index) * 64,
                    "evaluation_record_sha256": str(index + 3) * 64,
                    "trained_energy_checkpoint_sha256": "1" * 64,
                }
                for index, seed in enumerate((1701, 1702, 1703), start=1)
            ],
        }
    else:
        record["portability_evidence"] = {
            "record_sha256": "7" * 64,
            "training_seed": 1701,
            "training_record_sha256": "1" * 64,
            "trained_energy_checkpoint_sha256": "1" * 64,
            "acceptance": "pass",
        }
    return record


def _bundle(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    environment_lock = {
        "schema": "flagquantum.qboson_qdiffusion_environment_lock",
        "version": "1.0",
        "inventory_policy": "exact",
        "python_version": "3.10.18",
        "distributions": [
            {
                "name": "kaiwu",
                "version": "1.3.1",
                "approved_artifact_sha256": "7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455",
                "installed_content_sha256": "c" * 64,
            },
            {
                "name": "torch",
                "version": "2.7.0",
                "approved_artifact_sha256": "a" * 64,
                "installed_content_sha256": "b" * 64,
            }
        ],
    }
    environment_path = tmp_path / "environment-lock.json"
    environment_hash = _write_json(environment_path, environment_lock)
    config = _config()
    config["software"]["environment_lock_sha256"] = environment_hash
    config_path = tmp_path / "config.json"
    config_hash = _write_json(config_path, config)
    records = [
        _record("jp-a800-171", "primary", config_hash, environment_hash),
        _record("jp-a800-172", "portability_replay", config_hash, environment_hash),
    ]
    entries = []
    for record in records:
        path = tmp_path / f"{record['execution_host']}.json"
        entries.append({"path": path.name, "sha256": _write_json(path, record)})
    manifest = {
        "schema": "flagquantum.qboson_qdiffusion_manifest",
        "version": "1.0",
        "config": {"path": config_path.name, "sha256": config_hash},
        "environment_lock": {
            "path": environment_path.name,
            "sha256": environment_hash,
        },
        "records": entries,
        "component_records": [],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, records


def _replace_record(manifest_path: Path, index: int, record: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["records"][index]["path"]
    manifest["records"][index]["sha256"] = _write_json(path, record)
    _write_json(manifest_path, manifest)


def _source_preflight(host: str, manifest_sha256: str) -> dict[str, Any]:
    revisions = (
        ("flagquantum-qboson-", "FlagQuantum-", _REVISION),
        ("kaiwu-plugin-", "kaiwu-pytorch-plugin-", _PLUGIN_REVISION),
        (
            "kaiwu-community-",
            "kaiwu-community-",
            "b648b531c034bd6ae9b7a34fed994c717967cc72",
        ),
    )
    return {
        "schema": "flagquantum.qboson_a800_extracted_bundle_verification",
        "version": "1.0",
        "evidence_class": "extraction_preflight_only",
        "verification_hostname": f"hostname-{host}",
        "verified_for_target_host": host,
        "manifest_sha256": manifest_sha256,
        "extracted_content_verified": True,
        "artifacts": [
            {
                "filename": f"{filename_prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "extracted_root": f"{root_prefix}{revision[:10]}",
                "file_count": 1,
                "content_set_sha256": "f" * 64,
            }
            for filename_prefix, root_prefix, revision in revisions
        ],
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }


def _transfer_manifest() -> dict[str, Any]:
    revisions = (
        ("flagquantum-qboson-", _REVISION),
        ("kaiwu-plugin-", _PLUGIN_REVISION),
        ("kaiwu-community-", "b648b531c034bd6ae9b7a34fed994c717967cc72"),
    )
    return {
        "schema": "flagquantum.qboson_a800_transfer_bundle",
        "version": "1.0",
        "created_for_hosts": ["jp-a800-171", "jp-a800-172"],
        "classification": "local_preparation_only_not_execution_evidence",
        "artifacts": [
            {
                "filename": f"{prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "sha256": "f" * 64,
            }
            for prefix, revision in revisions
        ],
    }


def test_host_only_manifest_cannot_bypass_required_component_bundle(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _bundle(tmp_path)

    errors = validate_acceptance(manifest_path)

    assert any("does not contain every source record" in error for error in errors)
    assert any("do not reference the exact component bundle" in error for error in errors)


def test_validator_rejects_foreign_owned_manifest_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _bundle(tmp_path)
    effective_uid = os.geteuid()
    monkeypatch.setattr(
        validator_module.os, "geteuid", lambda: effective_uid + 1
    )

    assert validate_acceptance(manifest_path) == [
        "manifest: must be owned by the current effective user"
    ]


@pytest.mark.parametrize(
    "location",
    ("manifest", "config", "environment_lock", "record", "component_record"),
)
def test_manifest_and_member_reference_field_sets_are_closed(
    tmp_path: Path, location: str
) -> None:
    manifest_path, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location == "manifest":
        manifest["unexpected_secret_field"] = "must-not-pass"
    elif location in {"config", "environment_lock"}:
        manifest[location]["unexpected_secret_field"] = "must-not-pass"
    elif location == "record":
        manifest["records"][0]["unexpected_secret_field"] = "must-not-pass"
    else:
        component_path = tmp_path / "component.json"
        component_sha = _write_json(component_path, {"schema": "test.component"})
        manifest["component_records"] = [
            {
                "path": component_path.name,
                "sha256": component_sha,
                "unexpected_secret_field": "must-not-pass",
            }
        ]
    _write_json(manifest_path, manifest)

    errors = validate_acceptance(manifest_path)

    expected_label = {
        "manifest": "manifest:",
        "config": "manifest.config:",
        "environment_lock": "manifest.environment_lock:",
        "record": "manifest.records[0]:",
        "component_record": "manifest.component_records[0]:",
    }[location]
    assert any(
        error.startswith(expected_label) and "closed schema" in error
        for error in errors
    )


def test_validator_reads_each_evidence_member_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _bundle(tmp_path)
    from examples.qdiffusion_kaiwu import validate_acceptance as validator

    original = validator.read_private_bytes
    reads: list[Path] = []

    def counted_read(path: Path, *, label: str, max_bytes: int) -> bytes:
        reads.append(path)
        return original(path, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(validator, "read_private_bytes", counted_read)

    assert validator.validate_acceptance(manifest_path)
    assert len(reads) == len(set(reads)) == 5


def test_validator_requires_private_real_manifest_parent(tmp_path: Path) -> None:
    private_parent = tmp_path / "bundle"
    private_parent.mkdir(mode=0o700)
    manifest_path, _ = _bundle(private_parent)
    private_parent.chmod(0o755)

    errors = validate_acceptance(manifest_path)

    assert any("manifest" in error and "parent" in error for error in errors)


def test_validator_rejects_symlinked_or_public_bundle_members(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = tmp_path / manifest["records"][0]["path"]
    link = tmp_path / "linked-primary.json"
    link.symlink_to(original)
    manifest["records"][0]["path"] = link.name
    _write_json(manifest_path, manifest)

    assert any(
        "path contains a symlink" in error
        for error in validate_acceptance(manifest_path)
    )

    manifest["records"][0]["path"] = original.name
    _write_json(manifest_path, manifest)
    original.chmod(0o644)
    assert any(
        "accessible by group or others" in error
        for error in validate_acceptance(manifest_path)
    )


def test_validator_rejects_extra_or_duplicate_declared_members(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    _write_json(tmp_path / "unlisted.json", {"not": "evidence"})
    assert any(
        "exact declared member set" in error
        for error in validate_acceptance(manifest_path)
    )

    (tmp_path / "unlisted.json").unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][1]["path"] = manifest["records"][0]["path"]
    _write_json(manifest_path, manifest)
    assert any(
        "path is declared more than once" in error
        for error in validate_acceptance(manifest_path)
    )


def test_validator_rejects_extra_or_public_evidence_directories(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _bundle(tmp_path)
    extra = tmp_path / "unlisted-empty-directory"
    extra.mkdir(mode=0o700)
    assert any(
        "directories differ from the exact declared set" in error
        for error in validate_acceptance(manifest_path)
    )

    extra.rmdir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component = tmp_path / "components" / "component.json"
    component.parent.mkdir(mode=0o700)
    component_hash = _write_json(component, {"schema": "test.component"})
    manifest["component_records"] = [
        {"path": "components/component.json", "sha256": component_hash}
    ]
    _write_json(manifest_path, manifest)
    component.parent.chmod(0o755)

    errors = validate_acceptance(manifest_path)

    assert any("component record 0 parent" in error for error in errors)


def test_component_validator_rejects_different_host_transfer_manifests() -> None:
    config = _config()
    component_payloads = {
        "0" * 64: {
            "schema": "flagquantum.qboson_qdiffusion_artifact_preflight",
            "version": "1.0",
            "offline_preflight_only": True,
            "acceptance_evidence": False,
            "config_sha256": "c" * 64,
            "artifacts": {
                artifact_name: {
                    "sha256": config[config_name]["sha256"],
                    "algorithm": "file-sha256-v1",
                    "file_count": 1,
                }
                for artifact_name, config_name in {
                    "dataset": "dataset",
                    "base_checkpoint": "checkpoint",
                    "tokenizer": "tokenizer",
                    "evaluation_model": "evaluation_model",
                }.items()
            },
        },
        "1" * 64: _source_preflight("jp-a800-171", "a" * 64),
        "2" * 64: _source_preflight("jp-a800-172", "b" * 64),
        "a" * 64: _transfer_manifest(),
        "d" * 64: config["kaiwu_sdk"],
        "c" * 64: {
            "schema": "flagquantum.qboson_provider_resources",
            "version": "1.0",
            "source": "authenticated_resource_bill",
            "captured_at": "2026-10-05T12:00:00+00:00",
            "valid_until": "2026-10-06T12:00:00+00:00",
            "resources": [
                {
                    "target": target,
                    "mode": mode,
                    "available": 1,
                    "used": 0,
                }
                for target in ("SPQC-1", "SPQC-550", "SPQC-1000")
                for mode in ("optimization", "sampling")
            ],
            "claim_boundary": (
                "Account-resource observation only; it is not spend approval, "
                "project assignment, provider evidence, execution evidence, or "
                "acceptance evidence."
            ),
        },
        "e" * 64: {
            "schema": "flagquantum.qboson_kaiwu_live_smoke",
            "version": "1.0",
            "recorded_at": "2026-10-06T00:00:00+00:00",
            "transport": "kaiwu_cim",
            "real_provider_evidence": True,
            "qboson_hardware_used": True,
            "qboson_target": "SPQC-provider",
            "project_no": "CPQC-test",
            "environment_lock_sha256": config["software"][
                "environment_lock_sha256"
            ],
            "sdk_approval_sha256": "d" * 64,
            "provider_resources_sha256": "c" * 64,
            "tasks": [
                {
                    "receipt_schema": "flagquantum.kaiwu-task.v1",
                    "task_name": f"smoke-{mode}",
                    "task_mode": mode,
                    "matrix_sha256": (
                        "0352923b6964d8a65fc742c5a5b251ab967d43e8c4db9e3ee3a0f2f2fa5b0487"
                    ),
                    "matrix_size": 2,
                    "requested_samples": config["requested_samples"],
                    "project_no": "CPQC-test",
                    "submitted_at": "2026-10-06T00:00:00+00:00",
                    "returned_samples": config["requested_samples"],
                    "samples": [
                        [1, -1] for _ in range(config["requested_samples"])
                    ],
                    "energies": [
                        2.0 for _ in range(config["requested_samples"])
                    ],
                    "provider_task_id": f"task-{mode}",
                    "provider_target": "SPQC-provider",
                    "raw_status": "completed",
                    "fallback_occurred": False,
                    "minimum_energy": 2.0,
                    "maximum_energy": 2.0,
                    "provider_task_id_available": True,
                    "provider_target_available": True,
                }
                for mode in ("optimization", "sampling")
            ],
            "run_completed": True,
            "failure": None,
            "live_provider_smoke_passed": True,
            "provider_identity_complete": True,
            "hardware_acceptance": True,
            "fallback_occurred": False,
            "secrets_redacted": True,
            "limitations": [
                "This smoke test does not execute QDiffusion or A800 tensor work.",
                "Hardware acceptance remains false without provider-reported task and target identities.",
                "This record does not establish performance, quantum advantage, or production maturity.",
            ],
        },
    }
    schemas = (
        "flagquantum.qboson_qdiffusion_system_live_probe",
        "flagquantum.qboson_qdiffusion_system_live_probe",
        "flagquantum.qboson_qdiffusion_portability_replay",
        *("flagquantum.qboson_qdiffusion_protein_training",) * 3,
        *("flagquantum.qboson_qdiffusion_protein_evaluation",) * 3,
    )
    for index, schema in enumerate(schemas, start=3):
        component_payloads[str(index) * 64] = {
            "schema": schema,
            "version": "1.0",
            "experiment_config_sha256": "c" * 64,
        }
    errors: list[str] = []

    _validate_component_bundle(
        component_payloads,
        config=config,
        config_sha256="c" * 64,
        primary={},
        replay={},
        errors=errors,
    )

    assert "manifest: source preflights do not share one transfer manifest" in errors


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("transport", "in_memory_fake", "transport is not kaiwu_cim"),
        ("qboson_hardware_used", False, "hardware use is not proven"),
        ("fallback_occurred", True, "fallback must be explicitly false"),
        ("observed_tensor_device", "cpu", "tensor work was not observed"),
        ("real_provider_evidence", False, "real provider evidence is absent"),
    ),
)
def test_system_gate_rejects_false_claims(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed[field] = value
    _replace_record(manifest_path, 0, changed)

    assert any(message in error for error in validate_acceptance(manifest_path))


def test_application_gate_is_recomputed_from_metrics(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["guided_metrics"]["mean_cosine_distance"] = 0.6
    changed["guided_metrics"]["uniqueness_ratio"] = 0.8
    changed["guided_metrics"]["repeat_ratio_ge4"] = 0.1
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("mean cosine distance did not improve" in error for error in errors)
    assert any("uniqueness is below" in error for error in errors)
    assert any("repeat ratio exceeds" in error for error in errors)


def test_manifest_requires_independent_host_task_ids(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[1])
    changed["qboson_task_ids"] = records[0]["qboson_task_ids"]
    _replace_record(manifest_path, 1, changed)

    assert any(
        "independent QBoson task identities" in error
        for error in validate_acceptance(manifest_path)
    )


def test_tampered_record_hash_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    record_path = tmp_path / manifest["records"][0]["path"]
    record_path.write_text(record_path.read_text() + " ")

    assert any(
        "SHA-256 mismatch" in error for error in validate_acceptance(manifest_path)
    )


def test_system_gate_requires_aggregate_precision_evidence(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["precision_policy"]["matrix_count"] = 1
    changed["precision_policy"]["scale_factor_min"] = 3.0
    changed["precision_policy"]["scale_factor_max"] = 2.0
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("fewer reports than remote calls" in error for error in errors)
    assert any("invalid scale-factor range" in error for error in errors)


def test_system_gate_requires_source_preflight_identity(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["source_preflight_sha256"] = "self-reported"
    _replace_record(manifest_path, 0, changed)

    assert any(
        "source_preflight_sha256: expected a SHA-256" in error
        for error in validate_acceptance(manifest_path)
    )


def test_system_gate_rejects_inconsistent_transfer_accounting(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    boundary = changed["transfer_accounting"]["sampler_boundaries"][0]
    boundary["input_device"] = "cuda:0"
    boundary["cache_hit"] = True
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("input_device: expected cpu" in error for error in errors)
    assert any(
        "transfer accounting differs from remote-call count" in error
        for error in errors
    )


def test_manifest_requires_same_trained_energy_checkpoint(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[1])
    changed["artifacts"]["trained_energy_checkpoint_sha256"] = "2" * 64
    _replace_record(manifest_path, 1, changed)

    assert any(
        "same trained energy checkpoint" in error
        for error in validate_acceptance(manifest_path)
    )


def test_environment_lock_is_frozen_and_retained(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["environment_lock_sha256"] = "0" * 64
    _replace_record(manifest_path, 0, changed)

    assert any(
        "environment_lock_sha256: differs from the frozen software lane" in error
        for error in validate_acceptance(manifest_path)
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_path = tmp_path / manifest["environment_lock"]["path"]
    lock_path.write_text("{}", encoding="utf-8")
    lock_path.chmod(0o600)
    assert any(
        "manifest.environment_lock: SHA-256 mismatch" in error
        for error in validate_acceptance(manifest_path)
    )


def test_config_requires_environment_lock_digest() -> None:
    config = _config()
    config["software"]["environment_lock_sha256"] = "not-a-digest"
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("environment_lock_sha256" in error for error in errors)


@pytest.mark.parametrize(
    "host_identities",
    (
        {"jp-a800-171": "node-a800-171"},
        {
            "jp-a800-171": "<required>",
            "jp-a800-172": "node-a800-172",
        },
        {
            "jp-a800-171": "same-node",
            "jp-a800-172": "same-node",
        },
    ),
)
def test_config_requires_two_distinct_frozen_host_identities(
    host_identities: dict[str, str],
) -> None:
    config = _config()
    config["host_identities"] = host_identities
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("config.host_identities" in error for error in errors)


@pytest.mark.parametrize(
    "section",
    (
        None,
        "host_identities",
        "software",
        "kaiwu_sdk",
        "dataset",
        "checkpoint",
        "tokenizer",
        "evaluation_model",
        "training",
        "generation",
        "evaluation",
        "precision_policy",
        "primary_metric",
        "thresholds",
    ),
)
def test_config_rejects_undeclared_fields_that_could_hide_secrets(
    section: str | None,
) -> None:
    config = _config()
    target = config if section is None else config[section]
    target["sdk_code"] = "must-not-be-retained"
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("field set" in error or "host_identities" in error for error in errors)


def test_closed_config_fields_match_documented_template() -> None:
    template_path = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "acceptance_config.example.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))

    assert set(template) == CONFIG_FIELDS
    for section, expected_fields in CONFIG_SECTION_FIELDS.items():
        assert set(template[section]) == expected_fields


@pytest.mark.parametrize("value", ("<required>", None, ["0.2.0"]))
def test_config_requires_frozen_flagquantum_version(value: object) -> None:
    config = _config()
    config["software"]["flagquantum_version"] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert "config.software.flagquantum_version: frozen value is required" in errors


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("sha256", "not-a-digest", "expected a SHA-256 digest"),
        ("rights_reviewed_at", "2026-10-06", "timezone-aware timestamp"),
        ("approval_reference", "<required>", "frozen value is required"),
        ("project_no", "<required>", "assigned project is required"),
        (
            "project_assignment_reviewed_at",
            "2026-10-06",
            "timezone-aware timestamp",
        ),
        (
            "project_assignment_reference",
            "<required>",
            "frozen value is required",
        ),
        ("organizational_use_approved", False, "explicit approval is required"),
        ("isolated_container_use_approved", False, "explicit approval is required"),
        ("host_staging_approved", False, "explicit approval is required"),
        ("adapter_distribution_approved", False, "explicit approval is required"),
        ("sdk_redistribution_policy", "redistribute", "reviewed 1.3.1 lane"),
    ),
)
def test_config_requires_approved_kaiwu_sdk_rights(
    field: str, value: object, message: str
) -> None:
    config = _config()
    config["kaiwu_sdk"][field] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any(message in error for error in errors)


def test_environment_lock_kaiwu_artifact_is_bound_to_approval(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    lock_path = tmp_path / manifest["environment_lock"]["path"]
    environment_lock = json.loads(lock_path.read_text())
    environment_lock["distributions"][0]["approved_artifact_sha256"] = "b" * 64
    manifest["environment_lock"]["sha256"] = _write_json(
        lock_path, environment_lock
    )
    config_path = tmp_path / manifest["config"]["path"]
    config = json.loads(config_path.read_text())
    config["software"]["environment_lock_sha256"] = manifest["environment_lock"][
        "sha256"
    ]
    manifest["config"]["sha256"] = _write_json(config_path, config)
    _write_json(manifest_path, manifest)

    errors = validate_acceptance(manifest_path)

    assert any("Kaiwu artifact differs from SDK approval" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("dataset", "source_url", "http://example.test/data", "expected an HTTPS URL"),
        ("dataset", "source_url", "https://[broken", "expected an HTTPS URL"),
        ("checkpoint", "license_id", "NOASSERTION", "approved license identifier"),
        ("tokenizer", "license_evidence_url", "<required>", "expected an HTTPS URL"),
        (
            "evaluation_model",
            "license_reviewed_at",
            "2026-10-05",
            "timezone-aware timestamp",
        ),
    ),
)
def test_config_rejects_unapproved_artifact_provenance(
    section: str, field: str, value: object, message: str
) -> None:
    config = _config()
    config[section][field] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        (
            "dataset",
            "split",
            "ad-hoc",
            "expected deterministic-shuffle-v1",
        ),
        (
            "generation",
            "sequence_count",
            31,
            "differs from the frozen test split",
        ),
        (
            "generation",
            "disable_resample",
            "false",
            "expected a boolean",
        ),
        ("evaluation", "pair_mode", "nearest", "expected order"),
    ),
)
def test_config_rejects_workflow_parameter_drift(
    section: str, field: str, value: object, message: str
) -> None:
    config = _config()
    config[section][field] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any(message in error for error in errors)


def test_config_rejects_protein_budget_below_worst_case_estimate() -> None:
    config = _config()
    config["training"]["remote_call_budget_per_seed"] = 71268
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("worst-case workflow estimate of 71269" in error for error in errors)


def test_config_rejects_portability_fixture_outside_frozen_lane() -> None:
    config = _config()
    config["generation"]["portability_training_seed"] = 9999
    config["generation"]["portability_fixture_index"] = 32
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("expected one frozen seed" in error for error in errors)
    assert any("outside the frozen sequence set" in error for error in errors)


def test_config_rejects_system_budget_below_portability_estimate() -> None:
    config = _config()
    config["remote_call_budget"] = 16
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("portability replay estimate of 17" in error for error in errors)


@pytest.mark.parametrize("value", (None, 9, 2001, 10.0))
def test_config_rejects_unfrozen_provider_sample_count(value: object) -> None:
    config = _config()
    config["requested_samples"] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("config.requested_samples" in error for error in errors)


def test_component_bundle_requirement_cannot_be_disabled(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["component_bundle_required"] = False
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("component bundle must be explicitly required" in error for error in errors)
    assert any("does not contain every source record" in error for error in errors)
    assert any(
        "do not reference the exact component bundle" in error for error in errors
    )


def test_application_evidence_must_cover_frozen_seed_order(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["application_evidence"]["records"].reverse()
    _replace_record(manifest_path, 0, changed)

    assert any(
        "seed order differs from config" in error
        for error in validate_acceptance(manifest_path)
    )
