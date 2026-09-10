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
from .directed_topology import DirectedCouplingMap
from .direction_legalization import (
    DirectionLegalizationError,
    DirectionLegalizationResult,
    legalize_directed_cx,
)
from .native_gate_legalization import (
    NativeGateLegalizationError,
    NativeGateLegalizationResult,
    legalize_native_gates,
)
from .operator_lowering import (
    DEFAULT_LOWERING_REGISTRY,
    LoweringCapability,
    OperatorLoweringRegistry,
    UnsupportedLoweringError,
)
from .routing import CouplingMap
from .schedule_legalization import (
    CircuitSchedule,
    ScheduleLegalizationError,
    schedule_circuit_dependencies,
)
from .topology_legalization import (
    TopologyLegalizationError,
    TopologyLegalizationResult,
    legalize_circuit_topology,
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
    topology_legalization: TopologyLegalizationResult | None
    native_gate_legalization: NativeGateLegalizationResult
    direction_legalization: DirectionLegalizationResult | None
    lowering_capabilities: tuple[LoweringCapability, ...]
    capability_match: CapabilityMatchResult
    schedule: CircuitSchedule
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
    native_gate_legalization: NativeGateLegalizationResult,
    topology_legalization: TopologyLegalizationResult | None,
    direction_legalization: DirectionLegalizationResult | None,
    schedule: CircuitSchedule,
) -> str:
    payload = {
        "backend": backend,
        "circuit_content_hash": ir.content_hash,
        "requirement_set_id": requirements.requirement_set_id,
        "target_snapshot_id": snapshot.snapshot_id,
        "native_gate_legalization_identity": (
            native_gate_legalization.legalization_identity
        ),
        "topology_legalization_identity": (
            None
            if topology_legalization is None
            else topology_legalization.legalization_identity
        ),
        "direction_legalization_identity": (
            None
            if direction_legalization is None
            else direction_legalization.legalization_identity
        ),
        "schedule_identity": schedule.schedule_identity,
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
    max_added_operations: int = 256,
    coupling_map: CouplingMap | DirectedCouplingMap | None = None,
    routing_strategy: str = "auto",
    max_routing_added_operations: int = 256,
    initial_layout: tuple[int, ...] | None = None,
    max_direction_added_operations: int = 256,
    max_schedule_depth: int | None = None,
) -> TargetLegalizationResult:
    """Require an exact backend lowering and a matching target snapshot.

    The accepted circuit remains a Core-owned ``CircuitIR`` and may contain
    verified native-gate decompositions. This function does not select a
    target, execute a program, authorize fallback, or create a target-specific
    IR.
    """

    source_ir = ensure_circuit_ir(program)
    topology_legalization = None
    if coupling_map is not None:
        try:
            topology_legalization = legalize_circuit_topology(
                source_ir,
                coupling_map=coupling_map,
                snapshot=snapshot,
                strategy=routing_strategy,
                max_added_operations=max_routing_added_operations,
                initial_layout=initial_layout,
            )
        except TopologyLegalizationError as error:
            raise TargetLegalizationError(str(error)) from error
        source_ir = topology_legalization.program
    try:
        native_gate_legalization = legalize_native_gates(
            source_ir,
            snapshot=snapshot,
            evaluated_at=evaluated_at,
            max_added_operations=max_added_operations,
        )
    except NativeGateLegalizationError as error:
        raise TargetLegalizationError(str(error)) from error
    ir = native_gate_legalization.program
    direction_legalization = None
    if isinstance(coupling_map, DirectedCouplingMap):
        try:
            direction_legalization = legalize_directed_cx(
                ir,
                coupling_map=coupling_map,
                snapshot=snapshot,
                native_opcodes=native_gate_legalization.native_opcodes,
                max_added_operations=max_direction_added_operations,
            )
        except DirectionLegalizationError as error:
            raise TargetLegalizationError(str(error)) from error
        ir = direction_legalization.program
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

    try:
        schedule = schedule_circuit_dependencies(
            ir,
            target_snapshot_id=snapshot.snapshot_id,
            max_depth=max_schedule_depth,
        )
    except ScheduleLegalizationError as error:
        raise TargetLegalizationError(str(error)) from error

    return TargetLegalizationResult(
        program=ir,
        backend=normalized_backend,
        requirements=requirements,
        target_snapshot_id=snapshot.snapshot_id,
        topology_legalization=topology_legalization,
        native_gate_legalization=native_gate_legalization,
        direction_legalization=direction_legalization,
        lowering_capabilities=lowerings,
        capability_match=match,
        schedule=schedule,
        legalization_identity=_legalization_identity(
            ir,
            normalized_backend,
            requirements,
            snapshot,
            lowerings,
            native_gate_legalization,
            topology_legalization,
            direction_legalization,
            schedule,
        ),
    )


__all__ = [
    "TargetLegalizationError",
    "TargetLegalizationResult",
    "circuit_target_requirements",
    "legalize_circuit_for_target",
]
