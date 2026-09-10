"""Capability-driven legality checks before target-specific emission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..core.target_capabilities import (
    CapabilityMatchResult,
    CapabilityRequirement,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    TargetCapabilitySnapshot,
    match_target_capabilities,
)
from ..errors import CompilationError
from .operator_lowering import (
    DEFAULT_LOWERING_REGISTRY,
    LoweringCapability,
    OperatorLoweringRegistry,
    UnsupportedLoweringError,
)


class TargetLegalizationError(CompilationError):
    """A valid circuit cannot be legalized for the requested target."""


@dataclass(frozen=True)
class TargetLegalizationResult:
    """Successful target-legality evidence without a second circuit IR."""

    program: CircuitIR
    backend: str
    requirements: RequirementSet
    target_snapshot_id: str
    lowering_capabilities: tuple[LoweringCapability, ...]
    capability_match: CapabilityMatchResult
    legalization_identity: str


def _requirement(
    name: str,
    operator: ComparisonOperator,
    value: object,
    *,
    observed: bool = False,
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name=name,
        operator=operator,
        value=value,
        strength=RequirementStrength.MANDATORY,
        source=RequirementSource.COMPILER,
        minimum_evidence_level=(
            EvidenceLevel.OBSERVABLE if observed else EvidenceLevel.BASIC
        ),
        accepted_exposures=(
            (FactExposure.OBSERVED,)
            if observed
            else (FactExposure.DECLARED, FactExposure.OBSERVED)
        ),
    )


def circuit_target_requirements(program: object) -> RequirementSet:
    """Derive target requirements that are authoritative in ``CircuitIR``."""

    ir = ensure_circuit_ir(program)
    if ir.dtype not in {"complex64", "complex128"}:
        raise TargetLegalizationError(
            f"target legalization does not support CircuitIR dtype {ir.dtype!r}"
        )

    requirements = [
        _requirement(
            "qubits.logical_capacity",
            ComparisonOperator.AT_LEAST,
            ir.n_wires,
        ),
        _requirement(
            "limits.maximum_program_operations",
            ComparisonOperator.AT_LEAST,
            len(ir.instructions),
        ),
        _requirement(
            "precision.effective_dtype",
            ComparisonOperator.EQUALS,
            ir.dtype,
            observed=True,
        ),
    ]

    result_kinds = {
        str(item.metadata.get("fq_output_kind", item.kind)) for item in ir.measurements
    }
    if ir.observables:
        result_kinds.add("expectation")
    if result_kinds:
        requirements.append(
            _requirement(
                "measurements.results",
                ComparisonOperator.CONTAINS_ALL,
                tuple(sorted(result_kinds)),
            )
        )

    finite_shots = [item.shots for item in ir.measurements if item.shots is not None]
    if finite_shots:
        requirements.append(
            _requirement(
                "limits.maximum_shots",
                ComparisonOperator.AT_LEAST,
                max(finite_shots),
            )
        )
    return RequirementSet(requirements=tuple(requirements))


def _legalization_identity(
    ir: CircuitIR,
    backend: str,
    requirements: RequirementSet,
    snapshot: TargetCapabilitySnapshot,
    lowerings: tuple[LoweringCapability, ...],
) -> str:
    payload = {
        "backend": backend,
        "circuit_content_hash": ir.content_hash,
        "requirement_set_id": requirements.requirement_set_id,
        "target_snapshot_id": snapshot.snapshot_id,
        "lowerings": [
            {
                "opcode": item.opcode,
                "strategy": item.strategy,
                "implementation": item.implementation,
            }
            for item in lowerings
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def legalize_circuit_for_target(
    program: object,
    *,
    backend: str,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None = None,
    registry: OperatorLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> TargetLegalizationResult:
    """Require an exact backend lowering and a matching target snapshot.

    The accepted circuit remains the Core-owned ``CircuitIR``. This function
    emits legality evidence only; it does not select a target, execute a
    program, authorize fallback, or create a target-specific IR.
    """

    ir = ensure_circuit_ir(program)
    normalized_backend = str(backend).strip().lower()
    opcodes = tuple(sorted({item.name for item in ir.instructions}))
    try:
        lowerings = registry.validate(normalized_backend, opcodes)
    except (UnsupportedLoweringError, ValueError) as error:
        raise TargetLegalizationError(str(error)) from error

    requirements = circuit_target_requirements(ir)
    match = match_target_capabilities(
        requirements,
        snapshot,
        evaluated_at=evaluated_at,
    )
    if not match.executable:
        details = "; ".join(
            f"{item.capability_name or 'target'}: {item.code}: {item.message}"
            for item in match.blockers
        )
        raise TargetLegalizationError(
            f"target capability snapshot cannot legalize CircuitIR: {details}"
        )

    return TargetLegalizationResult(
        program=ir,
        backend=normalized_backend,
        requirements=requirements,
        target_snapshot_id=snapshot.snapshot_id,
        lowering_capabilities=lowerings,
        capability_match=match,
        legalization_identity=_legalization_identity(
            ir, normalized_backend, requirements, snapshot, lowerings
        ),
    )


__all__ = [
    "TargetLegalizationError",
    "TargetLegalizationResult",
    "circuit_target_requirements",
    "legalize_circuit_for_target",
]
