"""Loss-accounted adapter from Compiler target capabilities to Core v1.

The legacy :class:`TargetCapabilities` object remains authoritative for all
Compiler legality decisions.  This module only projects its closed, static
subset into the internal Core Target Capabilities v1 vocabulary.  Fields or
comparison semantics that Core v1 cannot preserve are recorded explicitly and
continue to be checked by the legacy Compiler comparator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from flagquantum.core.target_capabilities import (
    CapabilityRequirement,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
)

from .capability_comparison import CapabilityComparison, compare_target_capabilities
from .target_capabilities import ControlFlowProfile, TargetCapabilities

COMPILER_TARGET_ADAPTER_VERSION = "1.0"
SUPPORTED_LEGACY_SCHEMA = "target_capabilities_v1"
ADAPTER_OWNER = "compiler"
ADAPTER_EXIT_CONDITION = (
    "Remove only after every Compiler consumer uses an approved Core contract "
    "and golden legacy schema, fingerprint, and compatibility fixtures still pass."
)


class CompilerProjectionLossCode(str, Enum):
    """Closed reasons why the Core v1 projection is not independently complete."""

    DEFERRED_FROM_CORE_V1 = "deferred_from_core_v1"
    LEGACY_COMPARATOR_REQUIRED = "legacy_comparator_required"


@dataclass(frozen=True, order=True)
class CompilerProjectionLoss:
    """One legacy semantic field retained under Compiler authority."""

    field: str
    code: CompilerProjectionLossCode
    semantic_value_json: str
    active: bool
    message: str
    authority: str = "compiler.TargetCapabilities"

    def __post_init__(self) -> None:
        if not self.field or not self.message or not self.authority:
            raise ValueError("compiler projection loss fields must be non-empty")
        if not isinstance(self.code, CompilerProjectionLossCode):
            raise ValueError("compiler projection loss code must use its closed enum")
        if type(self.active) is not bool:
            raise ValueError("compiler projection loss active flag must be boolean")
        try:
            json.loads(self.semantic_value_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("semantic_value_json must be valid JSON") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "code": self.code.value,
            "semantic_value": json.loads(self.semantic_value_json),
            "active": self.active,
            "message": self.message,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class CompilerRequirementProjection:
    """Core requirement projection plus an exact legacy recovery record."""

    requirement_set: RequirementSet
    legacy_semantic_json: str
    legacy_semantic_fingerprint: str
    losses: tuple[CompilerProjectionLoss, ...]

    def restore_legacy(self, *, display_label: str | None = None) -> TargetCapabilities:
        payload = json.loads(self.legacy_semantic_json)
        payload["display_label"] = display_label
        restored = TargetCapabilities.from_dict(payload)
        if restored.semantic_fingerprint != self.legacy_semantic_fingerprint:
            raise ValueError("restored legacy target capability fingerprint changed")
        return restored

    def compare_available(self, available: TargetCapabilities) -> CapabilityComparison:
        """Apply the unchanged legacy coverage semantics to an available target."""

        if not isinstance(available, TargetCapabilities):
            raise TypeError("available must be a Compiler TargetCapabilities value")
        return compare_target_capabilities(self.restore_legacy(), available)

    @property
    def requires_legacy_comparator(self) -> bool:
        """The Core subset is never a complete Compiler legality verdict."""

        return any(
            loss.code is CompilerProjectionLossCode.LEGACY_COMPARATOR_REQUIRED
            or loss.active
            for loss in self.losses
        )


_DIRECT_FIELDS = frozenset(
    {
        "target_class",
        "logical_qubit_capacity",
        "physical_qubit_capacity",
        "native_gates",
        "measurement_results",
        "artifact_profiles",
        "maximum_shots",
        "maximum_program_operations",
        "ancilla_policy",
        "maximum_compiler_ancillas",
    }
)
_DEFERRED_FIELDS = (
    "topology",
    "control_flow",
    "supports_mid_circuit_measurement",
    "supports_reset",
    "supports_timing",
    "supports_pulse",
    "supports_noise",
    "supports_parameter_binding",
    "calibration_snapshot_hash",
    "calibration_valid_until",
)
_LEGACY_COMPARATOR_FIELDS = ("native_gates", "ancilla_policy")
_SEMANTIC_FIELDS = frozenset(
    {
        "target_class",
        "logical_qubit_capacity",
        "physical_qubit_capacity",
        "native_gates",
        "measurement_results",
        "artifact_profiles",
        "topology",
        "control_flow",
        "supports_mid_circuit_measurement",
        "supports_reset",
        "supports_timing",
        "supports_pulse",
        "supports_noise",
        "supports_parameter_binding",
        "maximum_shots",
        "maximum_program_operations",
        "ancilla_policy",
        "maximum_compiler_ancillas",
        "calibration_snapshot_hash",
        "calibration_valid_until",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _legacy_record(target: TargetCapabilities) -> tuple[str, str]:
    payload = target.semantic_dict()
    if payload["schema_version"] != SUPPORTED_LEGACY_SCHEMA:
        raise ValueError("unsupported legacy target capability schema")
    semantic_json = _canonical_json(payload)
    return semantic_json, target.semantic_fingerprint


def _losses(target: TargetCapabilities) -> tuple[CompilerProjectionLoss, ...]:
    payload = target.semantic_dict()
    losses: list[CompilerProjectionLoss] = []
    for field in _DEFERRED_FIELDS:
        value = payload[field]
        if field == "control_flow":
            active = value != ControlFlowProfile.STATIC_ONLY.value
        elif field.startswith("supports_"):
            active = value is True
        else:
            active = value is not None
        losses.append(
            CompilerProjectionLoss(
                field=field,
                code=CompilerProjectionLossCode.DEFERRED_FROM_CORE_V1,
                semantic_value_json=_canonical_json(value),
                active=active,
                message=(
                    f"{field} is deferred from Core Target Capabilities v1 and "
                    "remains under the legacy Compiler comparator"
                ),
            )
        )
    for field in _LEGACY_COMPARATOR_FIELDS:
        losses.append(
            CompilerProjectionLoss(
                field=field,
                code=CompilerProjectionLossCode.LEGACY_COMPARATOR_REQUIRED,
                semantic_value_json=_canonical_json(payload[field]),
                active=True,
                message=(
                    f"{field} has Compiler-owned coverage semantics richer than "
                    "the conservative Core v1 generic comparator"
                ),
            )
        )
    return tuple(sorted(losses))


def _requirement(
    name: str,
    operator: ComparisonOperator,
    value: object,
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name=name,
        operator=operator,
        value=value,
        strength=RequirementStrength.MANDATORY,
        source=RequirementSource.COMPILER,
        minimum_evidence_level=EvidenceLevel.BASIC,
        accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
    )


def _projected_values(target: TargetCapabilities) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = [
        ("target.class", target.target_class.value),
        ("qubits.logical_capacity", target.logical_qubit_capacity),
        ("qubits.physical_capacity", target.physical_qubit_capacity),
        ("measurements.results", [item.value for item in target.measurement_results]),
        ("artifacts.profiles", [item.to_dict() for item in target.artifact_profiles]),
        ("ancillas.maximum_compiler", target.maximum_compiler_ancillas),
    ]
    if target.maximum_shots is not None:
        values.append(("limits.maximum_shots", target.maximum_shots))
    if target.maximum_program_operations is not None:
        values.append(
            ("limits.maximum_program_operations", target.maximum_program_operations)
        )
    return tuple(values)


def target_capabilities_to_requirement_set(
    target: TargetCapabilities,
) -> CompilerRequirementProjection:
    """Project one legacy Compiler requirement without changing its authority."""

    if not isinstance(target, TargetCapabilities):
        raise TypeError("target must be a Compiler TargetCapabilities value")
    operators = {
        "target.class": ComparisonOperator.EQUALS,
        "qubits.logical_capacity": ComparisonOperator.AT_LEAST,
        "qubits.physical_capacity": ComparisonOperator.AT_LEAST,
        "measurements.results": ComparisonOperator.CONTAINS_ALL,
        "artifacts.profiles": ComparisonOperator.CONTAINS_ALL,
        "limits.maximum_shots": ComparisonOperator.AT_LEAST,
        "limits.maximum_program_operations": ComparisonOperator.AT_LEAST,
        "ancillas.maximum_compiler": ComparisonOperator.AT_LEAST,
    }
    requirements = tuple(
        _requirement(name, operators[name], value)
        for name, value in _projected_values(target)
    )
    semantic_json, fingerprint = _legacy_record(target)
    return CompilerRequirementProjection(
        requirement_set=RequirementSet(requirements=requirements),
        legacy_semantic_json=semantic_json,
        legacy_semantic_fingerprint=fingerprint,
        losses=_losses(target),
    )


def legacy_semantic_field_accounting() -> dict[str, str]:
    """Return the complete static accounting used by contract tests."""

    accounting = {field: "projected" for field in _DIRECT_FIELDS}
    accounting.update({field: "deferred" for field in _DEFERRED_FIELDS})
    accounting.update(
        {
            field: "projected_with_legacy_comparator"
            for field in _LEGACY_COMPARATOR_FIELDS
        }
    )
    if set(accounting) != _SEMANTIC_FIELDS:
        raise AssertionError("legacy semantic field accounting is incomplete")
    return accounting


__all__ = [
    "ADAPTER_EXIT_CONDITION",
    "ADAPTER_OWNER",
    "COMPILER_TARGET_ADAPTER_VERSION",
    "SUPPORTED_LEGACY_SCHEMA",
    "CompilerProjectionLoss",
    "CompilerProjectionLossCode",
    "CompilerRequirementProjection",
    "legacy_semantic_field_accounting",
    "target_capabilities_to_requirement_set",
]
