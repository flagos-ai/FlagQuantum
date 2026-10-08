from __future__ import annotations

import copy
from typing import Any

import pytest

from examples.qdiffusion_kaiwu.provider_reconciliation import (
    CLAIM_BOUNDARY,
    FIELDS,
    SCHEMA,
    VERSION,
    apply_provider_reconciliations,
    validate_provider_reconciliation_record,
    validate_reconciliation_binding,
)

pytestmark = pytest.mark.unit


def _receipt() -> dict[str, Any]:
    return {
        "task_name": "flagquantum-sampling-20261008",
        "mode": "sampling",
        "matrix_sha256": "a" * 64,
        "requested_samples": 10,
        "submitted_at": "2026-10-08T03:34:36+00:00",
        "provider_task_id": None,
        "provider_target": None,
    }


def _record() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "captured_at": "2026-10-08T03:59:26+00:00",
        "source": "authenticated_resource_bill",
        "component_record_sha256": "b" * 64,
        "local_task_name": "flagquantum-sampling-20261008",
        "local_submitted_at": "2026-10-08T03:34:36+00:00",
        "matrix_sha256": "a" * 64,
        "mode": "sampling",
        "requested_samples": 10,
        "provider_batch_id": "S-XIAS-S63M-YC9I-E9L1",
        "associated_task_id": "0982518e44654747bfc4f45e87f950d1",
        "provider_target": "SPQC-1000",
        "transaction_channel": "SDK Create Task",
        "transaction_type": "Consumption",
        "resource_delta": -10,
        "unique_account_match": True,
        "hardware_acceptance": False,
        "qdiffusion_acceptance": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def test_reconciliation_contract_is_closed_and_binds_missing_runtime_identity() -> None:
    record = _record()

    assert set(record) == FIELDS
    assert validate_provider_reconciliation_record(record) == []
    assert (
        validate_reconciliation_binding(
            record,
            component_record_sha256="b" * 64,
            receipt=_receipt(),
        )
        == []
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("unique_account_match", False, "unique account match is not attested"),
        ("resource_delta", -9, "resource_delta does not match the task request"),
        (
            "hardware_acceptance",
            True,
            "standalone hardware acceptance must remain false",
        ),
        (
            "qdiffusion_acceptance",
            True,
            "standalone QDiffusion acceptance must remain false",
        ),
        (
            "provider_target",
            " SPQC-1000",
            "provider_target is not a canonical identifier",
        ),
        (
            "associated_task_id",
            "task\nid",
            "associated_task_id is not a canonical identifier",
        ),
        (
            "captured_at",
            "2026-10-07T03:59:26+00:00",
            "evidence capture predates local submission",
        ),
        (
            "captured_at",
            "2026-10-16T03:59:26+00:00",
            "evidence capture exceeds the seven-day bound",
        ),
    ),
)
def test_reconciliation_rejects_untrusted_or_ambiguous_evidence(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _record()
    record[field] = value

    assert any(
        message in error for error in validate_provider_reconciliation_record(record)
    )


def test_reconciliation_rejects_schema_extensions() -> None:
    record = _record()
    record["raw_provider_payload"] = {"secret": "must-not-pass"}

    assert validate_provider_reconciliation_record(record)[0] == (
        "provider reconciliation: field set differs from its closed schema"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"local_task_name": "another-task"}, "local_task_name differs"),
        ({"matrix_sha256": "d" * 64}, "matrix_sha256 differs"),
        ({"mode": "optimization", "resource_delta": -1}, "mode differs"),
        ({"requested_samples": 11, "resource_delta": -11}, "requested_samples differs"),
        (
            {"local_submitted_at": "2026-10-08T03:34:37+00:00"},
            "local_submitted_at differs",
        ),
        ({"component_record_sha256": "e" * 64}, "component record identity mismatch"),
    ),
)
def test_reconciliation_binding_rejects_cross_task_reuse(
    mutation: dict[str, object],
    message: str,
) -> None:
    record = _record()
    record.update(mutation)

    assert any(
        message in error
        for error in validate_reconciliation_binding(
            record,
            component_record_sha256="b" * 64,
            receipt=_receipt(),
        )
    )


def test_reconciliation_must_not_contradict_runtime_identity() -> None:
    receipt = copy.deepcopy(_receipt())
    receipt["provider_task_id"] = "runtime-task"
    receipt["provider_target"] = "SPQC-550"

    errors = validate_reconciliation_binding(
        _record(),
        component_record_sha256="b" * 64,
        receipt=receipt,
    )

    assert (
        "provider reconciliation: provider task ID contradicts the runtime receipt"
        in errors
    )
    assert (
        "provider reconciliation: provider target contradicts the runtime receipt"
        in errors
    )


def test_optimization_reconciliation_consumes_one_credit() -> None:
    record = _record()
    record.update({"mode": "optimization", "resource_delta": -1})
    receipt = _receipt()
    receipt["mode"] = "optimization"

    assert (
        validate_reconciliation_binding(
            record,
            component_record_sha256="b" * 64,
            receipt=receipt,
        )
        == []
    )


def test_reconciliation_materializes_identity_without_mutating_source() -> None:
    receipt = _receipt()
    component = {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "transport": "kaiwu_cim",
        "run_completed": True,
        "failure": None,
        "pinned_sdk_client": True,
        "task_receipts": [receipt],
        "provider_identity_complete": False,
        "provider_reported_target": False,
        "real_provider_evidence": False,
        "qboson_hardware_used": False,
        "qboson_target": None,
        "qboson_task_ids": [],
        "acceptance": {"system": "fail", "application": "not_run"},
    }

    derived = apply_provider_reconciliations(
        component,
        component_record_sha256="b" * 64,
        reconciliations=[_record()],
    )

    assert component["task_receipts"][0]["provider_task_id"] is None
    assert component["acceptance"]["system"] == "fail"
    assert derived["task_receipts"][0]["provider_task_id"] == (
        "0982518e44654747bfc4f45e87f950d1"
    )
    assert derived["qboson_task_ids"] == ["0982518e44654747bfc4f45e87f950d1"]
    assert derived["qboson_target"] == "SPQC-1000"
    assert derived["acceptance"]["system"] == "pass"
    assert derived["real_provider_evidence"] is True


def test_reconciliation_cannot_upgrade_injected_transport() -> None:
    component = {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "transport": "injected_test",
        "run_completed": True,
        "failure": None,
        "pinned_sdk_client": False,
        "task_receipts": [_receipt()],
        "provider_identity_complete": False,
        "provider_reported_target": False,
        "real_provider_evidence": False,
        "qboson_hardware_used": False,
        "qboson_target": None,
        "qboson_task_ids": [],
        "acceptance": {"system": "fail", "application": "not_run"},
    }

    derived = apply_provider_reconciliations(
        component,
        component_record_sha256="b" * 64,
        reconciliations=[_record()],
    )

    assert derived["provider_identity_complete"] is True
    assert derived["real_provider_evidence"] is False
    assert derived["qboson_hardware_used"] is False
    assert derived["acceptance"]["system"] == "fail"


def test_reconciliation_materialization_rejects_duplicate_task_matches() -> None:
    component = {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "task_receipts": [_receipt()],
    }

    with pytest.raises(ValueError, match="ambiguous"):
        apply_provider_reconciliations(
            component,
            component_record_sha256="b" * 64,
            reconciliations=[_record(), copy.deepcopy(_record())],
        )


def test_reconciliation_cannot_attach_to_an_unrelated_component() -> None:
    component = {"schema": "unrelated.schema", "task_receipts": [_receipt()]}

    with pytest.raises(ValueError, match="unsupported component schema"):
        apply_provider_reconciliations(
            component,
            component_record_sha256="b" * 64,
            reconciliations=[_record()],
        )


def test_smoke_identity_can_be_derived_from_two_unique_bill_rows() -> None:
    optimization_receipt = _receipt()
    optimization_receipt.update(
        {
            "task_name": "smoke-optimization",
            "task_mode": "optimization",
            "provider_task_id_available": False,
            "provider_target_available": False,
        }
    )
    optimization_receipt.pop("mode")
    sampling_receipt = _receipt()
    sampling_receipt.update(
        {
            "task_name": "smoke-sampling",
            "task_mode": "sampling",
            "provider_task_id_available": False,
            "provider_target_available": False,
        }
    )
    sampling_receipt.pop("mode")
    component = {
        "schema": "flagquantum.qboson_kaiwu_live_smoke",
        "transport": "kaiwu_cim",
        "run_completed": True,
        "failure": None,
        "live_provider_smoke_passed": True,
        "tasks": [optimization_receipt, sampling_receipt],
        "provider_identity_complete": False,
        "real_provider_evidence": False,
        "qboson_hardware_used": False,
        "qboson_target": None,
        "hardware_acceptance": False,
    }
    optimization = _record()
    optimization.update(
        {
            "local_task_name": "smoke-optimization",
            "mode": "optimization",
            "provider_batch_id": "O-BATCH-1",
            "associated_task_id": "optimization-task",
            "resource_delta": -1,
        }
    )
    sampling = _record()
    sampling.update(
        {
            "local_task_name": "smoke-sampling",
            "provider_batch_id": "S-BATCH-1",
            "associated_task_id": "sampling-task",
        }
    )

    derived = apply_provider_reconciliations(
        component,
        component_record_sha256="b" * 64,
        reconciliations=[optimization, sampling],
    )

    assert derived["hardware_acceptance"] is True
    assert derived["provider_identity_complete"] is True
    assert derived["qboson_target"] == "SPQC-1000"
    assert [task["provider_task_id"] for task in derived["tasks"]] == [
        "optimization-task",
        "sampling-task",
    ]
    assert all(task["provider_task_id_available"] for task in derived["tasks"])
