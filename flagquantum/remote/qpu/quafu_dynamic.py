"""Offline Quafu dynamic-capability fixture normalization.

The normalizer is intentionally private and accepts conformance fixtures only.
It performs no provider construction, credential lookup, discovery, or network
access and does not establish a hardware-support claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from ...core.target_capabilities import (
    CapabilityFact,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
)
from ...deployment.cloud import CloudBackendProfile
from ...runtime.dynamic.conformance import DynamicFeatureSet
from ...runtime.dynamic.remote_capabilities import dynamic_remote_snapshot

_FIXTURE_SCHEMA_VERSION = "1.0"
_FIXTURE_SCOPE = "conformance_fixture_only"
_QUAFU_DYNAMIC_EXTENSION = "org.flagquantum.quafu-dynamic-v1"
_DYNAMIC_FIELDS = {
    "mid_circuit_measurement",
    "reset",
    "repeated_measurement",
    "classical_feed_forward",
    "conditional_operations",
    "condition_width",
    "condition_values",
    "condition_disjunction",
    "mid_circuit_result_data",
}


def _closed_mapping(
    value: object,
    *,
    field_name: str,
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise ValueError(
            "quafu_dynamic_capability_evidence_unavailable:"
            + f"{field_name}.{missing[0]}"
        )
    if unknown:
        raise ValueError(f"unsupported Quafu fixture field: {field_name}.{unknown[0]}")
    return dict(value)


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    return value.strip()


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    return value


def _boolean(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    items = tuple(_text(item, field_name=field_name) for item in value)
    if not items:
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    return tuple(sorted(set(items)))


def _integer_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    items = tuple(value)
    if not items or any(type(item) is not int for item in items):
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    return tuple(sorted(set(items)))


def _timestamp(value: object, *, field_name: str) -> datetime:
    text = _text(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"quafu_dynamic_capability_evidence_unavailable:{field_name}"
        ) from error
    if parsed.tzinfo is None:
        raise ValueError(f"quafu_dynamic_capability_evidence_unavailable:{field_name}")
    return parsed


def _directed_edges(value: object, *, capacity: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            "quafu_dynamic_capability_evidence_unavailable:"
            "backend.directed_connectivity"
        )
    edges: list[tuple[int, int]] = []
    for edge in value:
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, (str, bytes))
            or len(edge) != 2
            or any(type(qubit) is not int for qubit in edge)
        ):
            raise ValueError(
                "quafu_dynamic_capability_evidence_unavailable:"
                "backend.directed_connectivity"
            )
        source, target = edge
        if source == target or not 0 <= source < capacity or not 0 <= target < capacity:
            raise ValueError(
                "quafu_dynamic_capability_evidence_unavailable:"
                "backend.directed_connectivity"
            )
        edges.append((source, target))
    if not edges:
        raise ValueError(
            "quafu_dynamic_capability_evidence_unavailable:"
            "backend.directed_connectivity"
        )
    return tuple(sorted(set(edges)))


def _quafu_dynamic_snapshot_from_fixture(
    payload: Mapping[str, Any],
) -> TargetCapabilitySnapshot:
    """Normalize one closed, offline Quafu conformance fixture."""

    root = _closed_mapping(
        payload,
        field_name="fixture",
        required={
            "schema_version",
            "attestation_scope",
            "provider",
            "backend",
            "program",
            "dynamic",
            "identity",
        },
    )
    if root["schema_version"] != _FIXTURE_SCHEMA_VERSION:
        raise ValueError("unsupported Quafu dynamic fixture schema version")
    if root["attestation_scope"] != _FIXTURE_SCOPE:
        raise ValueError("Quafu dynamic fixture must be conformance_fixture_only")
    if root["provider"] != "quafu":
        raise ValueError("Quafu dynamic fixture provider must be 'quafu'")

    backend = _closed_mapping(
        root["backend"],
        field_name="backend",
        required={
            "name",
            "logical_qubits",
            "physical_qubits",
            "native_operations",
            "directed_connectivity",
        },
    )
    logical_qubits = _positive_int(
        backend["logical_qubits"], field_name="backend.logical_qubits"
    )
    physical_qubits = _positive_int(
        backend["physical_qubits"], field_name="backend.physical_qubits"
    )
    if logical_qubits > physical_qubits:
        raise ValueError("Quafu logical capacity cannot exceed physical capacity")
    native_operations = _string_tuple(
        backend["native_operations"], field_name="backend.native_operations"
    )
    directed_connectivity = _directed_edges(
        backend["directed_connectivity"], capacity=physical_qubits
    )

    program = _closed_mapping(
        root["program"],
        field_name="program",
        required={"dialect", "version"},
    )
    dialect = _text(program["dialect"], field_name="program.dialect")
    version = _text(program["version"], field_name="program.version")

    dynamic = _closed_mapping(
        root["dynamic"], field_name="dynamic", required=_DYNAMIC_FIELDS
    )
    if not _boolean(
        dynamic["classical_feed_forward"],
        field_name="dynamic.classical_feed_forward",
    ):
        raise ValueError(
            "quafu_dynamic_capability_evidence_unavailable:"
            "dynamic.classical_feed_forward"
        )
    features = DynamicFeatureSet(
        name="quafu-offline-contract-fixture",
        mid_circuit_measurement=_boolean(
            dynamic["mid_circuit_measurement"],
            field_name="dynamic.mid_circuit_measurement",
        ),
        reset=_boolean(dynamic["reset"], field_name="dynamic.reset"),
        supported_conditional_gates=_string_tuple(
            dynamic["conditional_operations"],
            field_name="dynamic.conditional_operations",
        ),
        max_condition_bits=_positive_int(
            dynamic["condition_width"], field_name="dynamic.condition_width"
        ),
        condition_values=_integer_tuple(
            dynamic["condition_values"], field_name="dynamic.condition_values"
        ),
        supports_disjunctive_conditions=_boolean(
            dynamic["condition_disjunction"],
            field_name="dynamic.condition_disjunction",
        ),
        returns_mid_circuit_measurements=_boolean(
            dynamic["mid_circuit_result_data"],
            field_name="dynamic.mid_circuit_result_data",
        ),
        supports_repeated_measurement=_boolean(
            dynamic["repeated_measurement"],
            field_name="dynamic.repeated_measurement",
        ),
    )

    identity = _closed_mapping(
        root["identity"],
        field_name="identity",
        required={
            "provider_version",
            "target_revision",
            "environment_id",
            "captured_at",
            "valid_until",
            "evidence_id",
            "evidence_state",
        },
    )
    captured_at = _timestamp(identity["captured_at"], field_name="identity.captured_at")
    valid_until = _timestamp(identity["valid_until"], field_name="identity.valid_until")
    ttl = valid_until - captured_at
    if ttl.total_seconds() <= 0:
        raise ValueError("identity.valid_until must be later than identity.captured_at")
    evidence_state = _text(
        identity["evidence_state"], field_name="identity.evidence_state"
    )
    if evidence_state != "declared":
        raise ValueError("offline Quafu fixtures may carry declared evidence only")
    evidence_sha256 = hashlib.sha256(
        json.dumps(
            root,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    profile = CloudBackendProfile(
        provider="quafu",
        name=_text(backend["name"], field_name="backend.name"),
        n_qubits=logical_qubits,
        basis_gates=native_operations,
        supports_dynamic_circuits=True,
        dynamic_dialect=f"{dialect}-{version}",
        metadata={"attestation_scope": _FIXTURE_SCOPE},
    )
    snapshot = dynamic_remote_snapshot(
        profile,
        features=features,
        captured_at=captured_at,
        ttl=ttl,
        provider_version=_text(
            identity["provider_version"], field_name="identity.provider_version"
        ),
        target_revision=_text(
            identity["target_revision"], field_name="identity.target_revision"
        ),
        environment_id=_text(
            identity["environment_id"], field_name="identity.environment_id"
        ),
        evidence_id=_text(identity["evidence_id"], field_name="identity.evidence_id"),
        evidence_sha256=evidence_sha256,
        evidence_state=evidence_state,
    )
    source = FactSource(
        kind="provider_declared",
        ref=snapshot.evidence_refs[0].evidence_id,
    )
    extensions = snapshot.to_dict()["extensions"]
    extensions[_QUAFU_DYNAMIC_EXTENSION] = {
        "schema_version": "1.0",
        "attestation_scope": _FIXTURE_SCOPE,
        "program": {"dialect": dialect, "version": version},
        "directed_connectivity": [list(edge) for edge in directed_connectivity],
    }
    return replace(
        snapshot,
        snapshot_id="",
        facts=(
            *snapshot.facts,
            CapabilityFact(
                name="qubits.physical_capacity",
                value=physical_qubits,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=source,
            ),
            CapabilityFact(
                name="gates.native",
                value=native_operations,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=source,
            ),
        ),
        extensions=extensions,
    )


__all__: tuple[str, ...] = ()
