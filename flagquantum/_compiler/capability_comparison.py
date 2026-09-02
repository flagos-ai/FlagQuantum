"""Structured, fail-closed comparison of private target capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .diagnostics import Diagnostic, DiagnosticCode
from .target_capabilities import (
    AncillaPolicy,
    ControlFlowProfile,
    GateCapability,
    ParameterConstraint,
    TargetCapabilities,
)


@dataclass(frozen=True)
class CapabilityDifference:
    field: str
    required: object
    available: object
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "required": self.required,
            "available": self.available,
            "message": self.message,
        }


@dataclass(frozen=True)
class CapabilityComparison:
    differences: tuple[CapabilityDifference, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def compatible(self) -> bool:
        return not self.differences


def _difference(field: str, required: Any, available: Any) -> CapabilityDifference:
    return CapabilityDifference(
        field,
        required,
        available,
        f"target capability {field!r} does not satisfy the requirement",
    )


def _constraint_covers(
    required: ParameterConstraint,
    available: ParameterConstraint,
) -> bool:
    if required.periodic and not available.periodic:
        return False
    if required.minimum is not None:
        if available.minimum is not None and available.minimum > required.minimum:
            return False
    if required.maximum is not None:
        if available.maximum is not None and available.maximum < required.maximum:
            return False
    return True


def _gate_covers(required: GateCapability, available: GateCapability) -> bool:
    if required.operation != available.operation:
        return False
    constraints = {item.name: item for item in available.parameters}
    required_constraints = {item.name: item for item in required.parameters}
    if set(constraints) - set(required_constraints):
        # The available target narrows a parameter that the requirement leaves open.
        return False
    for required_constraint in required.parameters:
        available_constraint = constraints.get(required_constraint.name)
        if available_constraint is None:
            # An omitted constraint means the full numeric domain is available.
            continue
        if not _constraint_covers(required_constraint, available_constraint):
            return False
    return True


def _append_set_difference(
    differences: list[CapabilityDifference],
    field: str,
    required: set[Any],
    available: set[Any],
) -> None:
    missing = required - available
    if missing:
        differences.append(_difference(field, sorted(required), sorted(available)))


def compare_target_capabilities(
    required: TargetCapabilities,
    available: TargetCapabilities,
) -> CapabilityComparison:
    """Return every semantic incompatibility; never silently degrade a requirement."""

    differences: list[CapabilityDifference] = []
    if required.target_class is not available.target_class:
        differences.append(
            _difference(
                "target_class",
                required.target_class.value,
                available.target_class.value,
            )
        )
    for field in ("logical_qubit_capacity", "physical_qubit_capacity"):
        if getattr(available, field) < getattr(required, field):
            differences.append(
                _difference(field, getattr(required, field), getattr(available, field))
            )
    available_gates = {item.operation: item for item in available.native_gates}
    missing_gates = [
        gate.to_dict()
        for gate in required.native_gates
        if gate.operation not in available_gates
        or not _gate_covers(gate, available_gates[gate.operation])
    ]
    if missing_gates:
        differences.append(
            _difference(
                "native_gates",
                missing_gates,
                [item.to_dict() for item in available.native_gates],
            )
        )
    _append_set_difference(
        differences,
        "measurement_results",
        {item.value for item in required.measurement_results},
        {item.value for item in available.measurement_results},
    )
    _append_set_difference(
        differences,
        "artifact_profiles",
        {(item.format.value, item.version) for item in required.artifact_profiles},
        {(item.format.value, item.version) for item in available.artifact_profiles},
    )
    if required.topology is not None:
        if available.topology is None:
            differences.append(_difference("topology", "explicit", None))
        else:
            _append_set_difference(
                differences,
                "topology.edges",
                set(required.topology.edges),
                set(available.topology.edges),
            )
    if (
        required.control_flow is ControlFlowProfile.ADAPTIVE_REAL_TIME
        and available.control_flow is not ControlFlowProfile.ADAPTIVE_REAL_TIME
    ):
        differences.append(
            _difference(
                "control_flow",
                required.control_flow.value,
                available.control_flow.value,
            )
        )
    for field in (
        "supports_mid_circuit_measurement",
        "supports_reset",
        "supports_timing",
        "supports_pulse",
        "supports_noise",
        "supports_parameter_binding",
    ):
        if getattr(required, field) and not getattr(available, field):
            differences.append(_difference(field, True, False))
    for field in ("maximum_shots", "maximum_program_operations"):
        required_limit = getattr(required, field)
        available_limit = getattr(available, field)
        if required_limit is not None and (
            available_limit is None or available_limit < required_limit
        ):
            differences.append(_difference(field, required_limit, available_limit))
    ancilla_order = {
        AncillaPolicy.UNSUPPORTED: 0,
        AncillaPolicy.EXPLICIT_ONLY: 1,
        AncillaPolicy.CLEAN_ALLOCATABLE: 2,
        AncillaPolicy.CLEAN_AND_DIRTY_ALLOCATABLE: 3,
    }
    if ancilla_order[available.ancilla_policy] < ancilla_order[required.ancilla_policy]:
        differences.append(
            _difference(
                "ancilla_policy",
                required.ancilla_policy.value,
                available.ancilla_policy.value,
            )
        )
    if available.maximum_compiler_ancillas < required.maximum_compiler_ancillas:
        differences.append(
            _difference(
                "maximum_compiler_ancillas",
                required.maximum_compiler_ancillas,
                available.maximum_compiler_ancillas,
            )
        )
    for field in ("calibration_snapshot_hash", "calibration_valid_until"):
        required_value = getattr(required, field)
        if required_value is not None and getattr(available, field) != required_value:
            differences.append(
                _difference(field, required_value, getattr(available, field))
            )
    diagnostics = tuple(
        Diagnostic(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            item.message,
            notes=(
                f"required={item.required!r}",
                f"available={item.available!r}",
            ),
        )
        for item in differences
    )
    return CapabilityComparison(tuple(differences), diagnostics)


__all__ = [
    "CapabilityComparison",
    "CapabilityDifference",
    "compare_target_capabilities",
]
