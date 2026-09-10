"""Legalize CircuitIR instructions against an evidenced native gate set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Mapping

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..core.target_capabilities import (
    CapabilityRequirement,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
    TargetCapabilitySnapshot,
    match_target_capabilities,
)
from ..errors import CompilationError


class NativeGateLegalizationError(CompilationError):
    """A circuit cannot be expressed by the evidenced native gate set."""


@dataclass(frozen=True)
class GateDecompositionRecord:
    """One deterministic source-to-native instruction replacement."""

    instruction_index: int
    source_opcode: str
    replacement_opcodes: tuple[str, ...]


@dataclass(frozen=True)
class NativeGateLegalizationResult:
    """A CircuitIR plus immutable native-gate legalization evidence."""

    program: CircuitIR
    source_content_hash: str
    target_snapshot_id: str
    native_opcodes: tuple[str, ...]
    decompositions: tuple[GateDecompositionRecord, ...]
    legalization_identity: str

    @property
    def changed(self) -> bool:
        return bool(self.decompositions)


@dataclass(frozen=True)
class _NativeGateDescriptor:
    opcode: str
    parameters: frozenset[str]

    def supports(self, instruction: Instruction) -> bool:
        schema = get_operator_schema(instruction.name)
        required = frozenset(schema.parameters if schema is not None else ())
        return required <= self.parameters


def _descriptor(raw: Any) -> _NativeGateDescriptor:
    if isinstance(raw, str):
        opcode = canonical_opcode(raw)
        schema = get_operator_schema(opcode)
        parameters = frozenset(schema.parameters if schema is not None else ())
        return _NativeGateDescriptor(opcode, parameters)
    if not isinstance(raw, Mapping):
        raise NativeGateLegalizationError(
            "gates.native entries must be opcode strings or descriptor objects"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise NativeGateLegalizationError(
            "gates.native descriptor requires a non-empty string name"
        )
    raw_parameters = raw.get("parameters", ())
    if isinstance(raw_parameters, (str, bytes)) or not isinstance(
        raw_parameters, (tuple, list)
    ):
        raise NativeGateLegalizationError(
            f"gates.native descriptor {name!r} parameters must be an array"
        )
    parameters = tuple(raw_parameters)
    if any(not isinstance(item, str) or not item for item in parameters):
        raise NativeGateLegalizationError(
            f"gates.native descriptor {name!r} parameters must be non-empty strings"
        )
    return _NativeGateDescriptor(canonical_opcode(name), frozenset(parameters))


def _native_descriptors(
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime | None,
) -> dict[str, tuple[_NativeGateDescriptor, ...]]:
    evidence_requirement = RequirementSet(
        requirements=(
            CapabilityRequirement(
                name="gates.native",
                operator=ComparisonOperator.CONTAINS_ALL,
                value=(),
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.COMPILER,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
            ),
        )
    )
    match = match_target_capabilities(
        evidence_requirement,
        snapshot,
        evaluated_at=evaluated_at,
    )
    if not match.executable:
        details = "; ".join(f"{item.code}: {item.message}" for item in match.blockers)
        raise NativeGateLegalizationError(
            f"target native-gate evidence is not usable: {details}"
        )
    fact = next(item for item in snapshot.facts if item.name == "gates.native")
    if fact.support_status is not SupportStatus.VERIFIED:
        raise AssertionError("Core capability matcher accepted an unverified gate fact")

    grouped: dict[str, list[_NativeGateDescriptor]] = {}
    for raw in fact.value:
        item = _descriptor(raw)
        variants = grouped.setdefault(item.opcode, [])
        if item not in variants:
            variants.append(item)
    if not grouped:
        raise NativeGateLegalizationError("target native gate set must not be empty")
    return {
        opcode: tuple(sorted(items, key=lambda item: tuple(sorted(item.parameters))))
        for opcode, items in grouped.items()
    }


def _supports(
    descriptors: Mapping[str, tuple[_NativeGateDescriptor, ...]],
    instruction: Instruction,
) -> bool:
    return any(
        item.supports(instruction) for item in descriptors.get(instruction.name, ())
    )


def _replacement(instruction: Instruction) -> tuple[Instruction, ...] | None:
    metadata = instruction.metadata
    wires = instruction.wires
    if instruction.name == "x":
        return (
            Instruction("h", wires, metadata=metadata),
            Instruction("z", wires, metadata=metadata),
            Instruction("h", wires, metadata=metadata),
        )
    if instruction.name == "rx":
        return (
            Instruction("h", wires, metadata=metadata),
            Instruction(
                "rz",
                wires,
                params={"theta": instruction.params["theta"]},
                metadata=metadata,
            ),
            Instruction("h", wires, metadata=metadata),
        )
    if instruction.name == "ry":
        return (
            Instruction("sdg", wires, metadata=metadata),
            Instruction("h", wires, metadata=metadata),
            Instruction(
                "rz",
                wires,
                params={"theta": instruction.params["theta"]},
                metadata=metadata,
            ),
            Instruction("h", wires, metadata=metadata),
            Instruction("s", wires, metadata=metadata),
        )
    if instruction.name == "swap":
        left, right = wires
        return (
            Instruction("cx", (left, right), metadata=metadata),
            Instruction("cx", (right, left), metadata=metadata),
            Instruction("cx", (left, right), metadata=metadata),
        )
    return None


def _identity(
    source: CircuitIR,
    result: CircuitIR,
    snapshot: TargetCapabilitySnapshot,
    native_opcodes: tuple[str, ...],
    decompositions: tuple[GateDecompositionRecord, ...],
) -> str:
    payload = {
        "source_content_hash": source.content_hash,
        "result_content_hash": result.content_hash,
        "target_snapshot_id": snapshot.snapshot_id,
        "native_opcodes": native_opcodes,
        "decompositions": [
            {
                "instruction_index": item.instruction_index,
                "source_opcode": item.source_opcode,
                "replacement_opcodes": item.replacement_opcodes,
            }
            for item in decompositions
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def legalize_native_gates(
    program: object,
    *,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None = None,
    max_added_operations: int = 256,
) -> NativeGateLegalizationResult:
    """Return an equivalent CircuitIR expressed in the target's native gates."""

    ir = ensure_circuit_ir(program)
    if not isinstance(max_added_operations, int) or isinstance(
        max_added_operations, bool
    ):
        raise TypeError("max_added_operations must be an integer")
    if max_added_operations < 0:
        raise ValueError("max_added_operations must be non-negative")
    descriptors = _native_descriptors(snapshot, evaluated_at=evaluated_at)

    instructions: list[Instruction] = []
    records: list[GateDecompositionRecord] = []
    for index, instruction in enumerate(ir.instructions):
        if _supports(descriptors, instruction):
            instructions.append(instruction)
            continue
        if instruction.matrix is not None:
            raise NativeGateLegalizationError(
                f"custom matrix instruction {instruction.name!r} has no native contract"
            )
        replacement = _replacement(instruction)
        if replacement is None:
            raise NativeGateLegalizationError(
                f"instruction {instruction.name!r} is not native and has no verified decomposition"
            )
        for item in replacement:
            if not _supports(descriptors, item):
                raise NativeGateLegalizationError(
                    f"decomposition of {instruction.name!r} requires unsupported native gate {item.name!r}"
                )
        added = len(instructions) + len(replacement) - index - 1
        if added > max_added_operations:
            raise NativeGateLegalizationError(
                "native-gate decomposition exceeds max_added_operations"
            )
        instructions.extend(replacement)
        records.append(
            GateDecompositionRecord(
                instruction_index=index,
                source_opcode=instruction.name,
                replacement_opcodes=tuple(item.name for item in replacement),
            )
        )

    result = replace(ir, instructions=tuple(instructions)) if records else ir
    native_opcodes = tuple(sorted(descriptors))
    decompositions = tuple(records)
    return NativeGateLegalizationResult(
        program=result,
        source_content_hash=ir.content_hash,
        target_snapshot_id=snapshot.snapshot_id,
        native_opcodes=native_opcodes,
        decompositions=decompositions,
        legalization_identity=_identity(
            ir, result, snapshot, native_opcodes, decompositions
        ),
    )


__all__ = [
    "GateDecompositionRecord",
    "NativeGateLegalizationError",
    "NativeGateLegalizationResult",
    "legalize_native_gates",
]
