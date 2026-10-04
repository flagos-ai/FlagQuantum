"""Legalize CircuitIR instructions against an evidenced native gate set.

A rewrite is chosen from two sources: the exact named-gate equivalences in
`basis_translation`, composed recursively until every leaf is native, and the
one-qubit Euler and two-qubit KAK syntheses for a matrix-carrying instruction,
which no operator schema describes. A synthesis applies only when the target
publishes the basis it needs: a z-rotation and a pi/2 x-rotation for one qubit,
plus a supercontrolled entangler for two. Among the rewrites that reach the
native set the shortest wins, so a basis that can carry a gate exactly keeps its
exact form instead of paying for the general one. A synthesized rewrite is equal
to its source only up to one global phase, which FlagQuantum IR cannot record;
see those modules for why.

A named two-qubit gate that is not native is translated through the equivalence
table rather than through a matrix decomposition, because the matrix of a named
gate belongs to `flagquantum.simulation`, which this layer must not import. A
named gate the table does not reach, and a matrix too wide for the entangler
basis, are refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from typing import Any

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
from .basis_translation import translate
from .one_qubit_synthesis import (
    HALF_PI_PULSE_OPCODES,
    Z_ROTATION_OPCODES,
    synthesize_one_qubit_matrix,
)
from .two_qubit_synthesis import SUPERCONTROLLED_ENTANGLERS, synthesize_two_qubit


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

    source_program: CircuitIR = field(repr=False)
    program: CircuitIR
    source_content_hash: str
    target_snapshot_id: str
    native_opcodes: tuple[str, ...]
    decompositions: tuple[GateDecompositionRecord, ...]
    legalization_identity: str

    def __post_init__(self) -> None:
        if self.source_program.content_hash != self.source_content_hash:
            raise ValueError(
                "native-gate source program does not match source_content_hash"
            )

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
        schema_parameters = frozenset(schema.parameters if schema is not None else ())
        return _NativeGateDescriptor(opcode, schema_parameters)
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


def _z_rotation_opcode(
    descriptors: Mapping[str, tuple[_NativeGateDescriptor, ...]],
) -> str | None:
    """Return the z-rotation the target publishes, or None when it has none."""
    for opcode in Z_ROTATION_OPCODES:
        if opcode in descriptors:
            return opcode
    return None


def _half_pi_pulse_opcode(
    descriptors: Mapping[str, tuple[_NativeGateDescriptor, ...]],
) -> str | None:
    """Return the pi/2 x-rotation the target publishes, or None."""
    for opcode in HALF_PI_PULSE_OPCODES:
        if opcode in descriptors:
            return opcode
    return None


def _entangler_opcode(
    descriptors: Mapping[str, tuple[_NativeGateDescriptor, ...]],
) -> str | None:
    """Return the supercontrolled entangler the target publishes, or None.

    The table is iterated rather than the target's descriptor map, because a
    target may publish several entanglers and the choice has to be a fixed
    function of the published set. `SUPERCONTROLLED_ENTANGLERS` is ordered so a
    parameter-free spelling is always preferred over one that has to be applied
    at an angle, and so the answer for a given target never depends on a mapping's
    insertion order.
    """

    for opcode in SUPERCONTROLLED_ENTANGLERS:
        if opcode in descriptors:
            return opcode
    return None


def _matrix_replacement(
    instruction: Instruction,
    *,
    z_rotation: str | None,
    pulse_opcode: str | None,
    entangler: str | None,
) -> tuple[Instruction, ...] | None:
    """Rewrite a matrix-carrying instruction, which no schema table describes.

    One qubit goes through Euler synthesis, two through KAK synthesis, and
    anything wider is refused because the entangler basis reaches exactly two.
    """

    if z_rotation is None or pulse_opcode is None:
        return None
    if len(instruction.wires) == 1:
        return synthesize_one_qubit_matrix(
            instruction.matrix,
            qubit=instruction.wires[0],
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=instruction.metadata,
        )
    if len(instruction.wires) == 2 and entangler is not None:
        return synthesize_two_qubit(
            instruction.matrix,
            qubits=instruction.wires,
            entangler=entangler,
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=instruction.metadata,
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
    z_rotation = _z_rotation_opcode(descriptors)
    pulse_opcode = _half_pi_pulse_opcode(descriptors)

    instructions: list[Instruction] = []
    records: list[GateDecompositionRecord] = []
    for index, instruction in enumerate(ir.instructions):
        if _supports(descriptors, instruction):
            instructions.append(instruction)
            continue
        if instruction.matrix is not None:
            replacement = _matrix_replacement(
                instruction,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                entangler=_entangler_opcode(descriptors),
            )
            if replacement is None:
                raise NativeGateLegalizationError(
                    f"custom matrix instruction {instruction.name!r} has no native contract"
                )
        else:
            replacement = translate(
                instruction,
                can_run=partial(_supports, descriptors),
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
            )
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
        source_program=ir,
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
