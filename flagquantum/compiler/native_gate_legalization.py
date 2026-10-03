"""Legalize CircuitIR instructions against an evidenced native gate set.

A rewrite is chosen from two sources: the exact named-gate equivalences in
`basis_translation`, composed recursively until every leaf is native, and the
one-qubit Euler and two-qubit KAK syntheses for a matrix-carrying instruction,
which no operator schema describes. A synthesis applies only when the target
publishes the basis it needs: a z-rotation and a pi/2 x-rotation for one wire,
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

The rewrite loop itself -- choose a rewrite, check every leaf, bound the added
operations, record the replacement -- is not here. It is `basis_conversion`'s
`convert_instructions`, because the same loop answers a second question that has
no device in it: convert this program from the basis it is written in into a
basis the caller names. What stays here is the part that is about a device:
reading `gates.native` out of the snapshot, matching it through the capability
matcher, and the wording a refusal uses, which `_REFUSALS` holds. A reader
looking for the search should follow the import.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
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

# `GateDecompositionRecord` is re-exported rather than defined here because the
# loop that produces it moved to `basis_conversion`; `remote/emulation.py`
# imports it from this module and is unchanged, which is the replacement this
# move had to pass. It is the same class object, so the two import paths agree.
from .basis_conversion import GateDecompositionRecord, convert_instructions


class NativeGateLegalizationError(CompilationError):
    """A circuit cannot be expressed by the evidenced native gate set."""


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


_REFUSALS = {
    "matrix": "custom matrix instruction {name!r} has no native contract",
    "no_rule": "instruction {name!r} is not native and has no verified decomposition",
    "escapes": ("decomposition of {name!r} requires unsupported native gate {gate!r}"),
    "budget": "native-gate decomposition exceeds max_added_operations",
}


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
    result, records = convert_instructions(
        ir,
        published=set(descriptors),
        can_run=partial(_supports, descriptors),
        refusals=_REFUSALS,
        error=NativeGateLegalizationError,
        max_added_operations=max_added_operations,
    )
    native_opcodes = tuple(sorted(descriptors))
    return NativeGateLegalizationResult(
        source_program=ir,
        program=result,
        source_content_hash=ir.content_hash,
        target_snapshot_id=snapshot.snapshot_id,
        native_opcodes=native_opcodes,
        decompositions=records,
        legalization_identity=_identity(ir, result, snapshot, native_opcodes, records),
    )


__all__ = [
    "GateDecompositionRecord",
    "NativeGateLegalizationError",
    "NativeGateLegalizationResult",
    "legalize_native_gates",
]
