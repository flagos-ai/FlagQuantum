"""Validate post-hoc QBoson Resource Bill task identity evidence.

Kaiwu 1.3.1 can complete a task without exposing the provider task ID or
provider target in its documented result mapping.  This module defines the
strict evidence record used to bind an immutable local component record to one
unique Resource Bill transaction without rewriting the original SDK receipt.
"""

from __future__ import annotations

import copy
import unicodedata
from datetime import datetime, timedelta
from typing import Any, TypedDict

from examples.qdiffusion_kaiwu.verify_environment_lock import SHA256

SCHEMA = "flagquantum.qboson_kaiwu_provider_reconciliation"
VERSION = "2.0"
SOURCE = "authenticated_resource_bill"
TRANSACTION_CHANNEL = "SDK Create Task"
TRANSACTION_TYPE = "Consumption"
CLAIM_BOUNDARY = (
    "Post-hoc account-bill identity evidence for one immutable local provider "
    "task; it is not a runtime SDK receipt and is not acceptance by itself."
)
MAX_CAPTURE_DELAY = timedelta(days=7)
_SMOKE_SCHEMA = "flagquantum.qboson_kaiwu_live_smoke"
_SYSTEM_SCHEMA = "flagquantum.qboson_qdiffusion_system_live_probe"
_TRAINING_SCHEMA = "flagquantum.qboson_qdiffusion_protein_training"
_PORTABILITY_SCHEMA = "flagquantum.qboson_qdiffusion_portability_replay"
SUPPORTED_COMPONENT_SCHEMAS = frozenset(
    {_SMOKE_SCHEMA, _SYSTEM_SCHEMA, _TRAINING_SCHEMA, _PORTABILITY_SCHEMA}
)


class ProviderReconciliationRecord(TypedDict):
    """Closed provider reconciliation schema."""

    schema: str
    version: str
    captured_at: str
    source: str
    component_record_sha256: str
    local_task_name: str
    local_submitted_at: str
    matrix_sha256: str
    mode: str
    requested_samples: int
    provider_batch_id: str
    associated_task_id: str
    provider_target: str
    transaction_channel: str
    transaction_type: str
    resource_delta: int
    unique_account_match: bool
    hardware_acceptance: bool
    qdiffusion_acceptance: bool
    claim_boundary: str


FIELDS = frozenset(ProviderReconciliationRecord.__annotations__)


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() != timedelta(0):
        return None
    return parsed


def _canonical_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and all(
            unicodedata.category(character) not in {"Cc", "Cf"} for character in value
        )
    )


def _expected_resource_delta(mode: object, requested_samples: object) -> int | None:
    if mode == "optimization":
        return -1
    if mode == "sampling" and type(requested_samples) is int and requested_samples > 0:
        return -requested_samples
    return None


def validate_provider_reconciliation_record(
    record: dict[str, Any],
    *,
    label: str = "provider reconciliation",
) -> list[str]:
    """Validate a standalone Resource Bill reconciliation record."""

    errors: list[str] = []
    if set(record) != FIELDS:
        errors.append(f"{label}: field set differs from its closed schema")
    fixed_values = {
        "schema": SCHEMA,
        "version": VERSION,
        "source": SOURCE,
        "transaction_channel": TRANSACTION_CHANNEL,
        "transaction_type": TRANSACTION_TYPE,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for field, expected in fixed_values.items():
        if record.get(field) != expected:
            errors.append(f"{label}: {field} differs from the fixed contract")
    for field in ("component_record_sha256", "matrix_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            errors.append(f"{label}: {field} is not a lowercase SHA-256 digest")
    for field in (
        "local_task_name",
        "provider_batch_id",
        "associated_task_id",
        "provider_target",
    ):
        if not _canonical_identifier(record.get(field)):
            errors.append(f"{label}: {field} is not a canonical identifier")
    if record.get("provider_batch_id") == record.get("associated_task_id"):
        errors.append(f"{label}: provider batch and task identities must differ")

    submitted_at = _parse_utc_timestamp(record.get("local_submitted_at"))
    captured_at = _parse_utc_timestamp(record.get("captured_at"))
    if submitted_at is None:
        errors.append(f"{label}: local_submitted_at is not an aware UTC timestamp")
    if captured_at is None:
        errors.append(f"{label}: captured_at is not an aware UTC timestamp")
    if submitted_at is not None and captured_at is not None:
        if captured_at < submitted_at:
            errors.append(f"{label}: evidence capture predates local submission")
        elif captured_at - submitted_at > MAX_CAPTURE_DELAY:
            errors.append(f"{label}: evidence capture exceeds the seven-day bound")

    mode = record.get("mode")
    requested_samples = record.get("requested_samples")
    if mode not in {"optimization", "sampling"}:
        errors.append(f"{label}: mode is unsupported")
    if type(requested_samples) is not int or requested_samples <= 0:
        errors.append(f"{label}: requested_samples must be a positive integer")
    expected_delta = _expected_resource_delta(mode, requested_samples)
    if expected_delta is not None and record.get("resource_delta") != expected_delta:
        errors.append(f"{label}: resource_delta does not match the task request")
    if record.get("unique_account_match") is not True:
        errors.append(f"{label}: unique account match is not attested")
    if record.get("hardware_acceptance") is not False:
        errors.append(f"{label}: standalone hardware acceptance must remain false")
    if record.get("qdiffusion_acceptance") is not False:
        errors.append(f"{label}: standalone QDiffusion acceptance must remain false")
    return errors


def validate_reconciliation_binding(
    record: dict[str, Any],
    *,
    component_record_sha256: str,
    receipt: dict[str, Any],
    label: str = "provider reconciliation",
) -> list[str]:
    """Validate that one reconciliation uniquely binds one immutable receipt."""

    errors = validate_provider_reconciliation_record(record, label=label)
    if record.get("component_record_sha256") != component_record_sha256:
        errors.append(f"{label}: component record identity mismatch")
    exact_bindings = {
        "local_task_name": receipt.get("task_name"),
        "matrix_sha256": receipt.get("matrix_sha256"),
        "mode": receipt.get("mode", receipt.get("task_mode")),
        "requested_samples": receipt.get("requested_samples"),
    }
    for field, expected in exact_bindings.items():
        if record.get(field) != expected:
            errors.append(f"{label}: {field} differs from the local receipt")
    local_submitted = _parse_utc_timestamp(record.get("local_submitted_at"))
    receipt_submitted = _parse_utc_timestamp(receipt.get("submitted_at"))
    if receipt_submitted is None:
        errors.append(f"{label}: local receipt submitted_at is invalid")
    elif local_submitted is not None and local_submitted != receipt_submitted:
        errors.append(f"{label}: local_submitted_at differs from the local receipt")

    runtime_task_id = receipt.get("provider_task_id")
    if runtime_task_id is not None and runtime_task_id != record.get(
        "associated_task_id"
    ):
        errors.append(f"{label}: provider task ID contradicts the runtime receipt")
    runtime_target = receipt.get("provider_target")
    if runtime_target is not None and runtime_target != record.get("provider_target"):
        errors.append(f"{label}: provider target contradicts the runtime receipt")
    return errors


def apply_provider_reconciliations(
    component: dict[str, Any],
    *,
    component_record_sha256: str,
    reconciliations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return an in-memory identity view without rewriting source evidence.

    The returned mapping is used only for acceptance derivation. The immutable
    original component and every reconciliation record remain separate bundle
    members and retain their own SHA-256 identities.
    """

    schema = component.get("schema")
    matching = [
        record
        for record in reconciliations
        if record.get("component_record_sha256") == component_record_sha256
    ]
    if schema not in SUPPORTED_COMPONENT_SCHEMAS:
        if matching:
            raise ValueError(
                "provider reconciliation references an unsupported component schema"
            )
        return copy.deepcopy(component)

    derived = copy.deepcopy(component)
    if not matching:
        return derived
    receipt_field = "tasks" if schema == _SMOKE_SCHEMA else "task_receipts"
    receipts = derived.get(receipt_field)
    if not isinstance(receipts, list) or not receipts:
        if matching:
            raise ValueError(
                "provider reconciliation references a component without task receipts"
            )
        return derived

    records_by_task: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(matching):
        errors = validate_provider_reconciliation_record(
            record, label=f"provider reconciliation {index}"
        )
        if errors:
            raise ValueError("; ".join(errors))
        task_name = record["local_task_name"]
        records_by_task.setdefault(task_name, []).append(record)

    used_record_ids: set[int] = set()
    effective_ids: list[str] = []
    effective_targets: set[str] = set()
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            continue
        task_name = receipt.get("task_name")
        task_matches = records_by_task.get(task_name, [])
        runtime_complete = _canonical_identifier(
            receipt.get("provider_task_id")
        ) and _canonical_identifier(receipt.get("provider_target"))
        if len(task_matches) > 1:
            raise ValueError(
                f"provider reconciliation is ambiguous for task {task_name!r}"
            )
        if task_matches:
            record = task_matches[0]
            binding_errors = validate_reconciliation_binding(
                record,
                component_record_sha256=component_record_sha256,
                receipt=receipt,
                label=f"provider reconciliation for task {task_name!r}",
            )
            if binding_errors:
                raise ValueError("; ".join(binding_errors))
            used_record_ids.add(id(record))
            receipt["provider_task_id"] = record["associated_task_id"]
            receipt["provider_target"] = record["provider_target"]
            if schema == _SMOKE_SCHEMA:
                receipt["provider_task_id_available"] = True
                receipt["provider_target_available"] = True
        elif not runtime_complete:
            continue
        effective_ids.append(receipt["provider_task_id"])
        effective_targets.add(receipt["provider_target"])

    if len(used_record_ids) != len(matching):
        raise ValueError("provider reconciliation does not identify one component task")
    target_assignment_valid = len(effective_targets) == 1 or (
        schema == _SMOKE_SCHEMA and derived.get("project_no") is None
    )
    identities_complete = (
        len(effective_ids) == len(receipts)
        and len(effective_ids) == len(set(effective_ids))
        and target_assignment_valid
    )
    if not identities_complete:
        return derived

    if schema == _SMOKE_SCHEMA:
        sampling_targets = {
            receipt.get("provider_target")
            for receipt in receipts
            if receipt.get("task_mode") == "sampling"
            and _canonical_identifier(receipt.get("provider_target"))
        }
        if len(sampling_targets) != 1:
            return derived
        target = next(iter(sampling_targets))
    else:
        target = next(iter(effective_targets))
    derived["provider_identity_complete"] = True
    derived["qboson_target"] = target
    provider_transport_proven = bool(
        derived.get("transport") == "kaiwu_cim"
        and derived.get("run_completed") is True
        and derived.get("failure") is None
        and (schema == _SMOKE_SCHEMA or derived.get("pinned_sdk_client") is True)
    )
    derived["real_provider_evidence"] = provider_transport_proven
    derived["qboson_hardware_used"] = provider_transport_proven
    if schema == _SMOKE_SCHEMA:
        derived["hardware_acceptance"] = bool(
            provider_transport_proven
            and derived.get("live_provider_smoke_passed") is True
        )
    else:
        derived["provider_reported_target"] = True
        derived["qboson_task_ids"] = effective_ids
        if schema == _SYSTEM_SCHEMA and provider_transport_proven:
            acceptance = derived.get("acceptance")
            if isinstance(acceptance, dict):
                acceptance["system"] = "pass"
        elif schema == _PORTABILITY_SCHEMA and provider_transport_proven:
            acceptance = derived.get("acceptance")
            if isinstance(acceptance, dict):
                acceptance["portability"] = "pass"
    return derived
