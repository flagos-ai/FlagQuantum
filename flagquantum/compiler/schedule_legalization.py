"""Dependency-preserving logical scheduling evidence for ``CircuitIR``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..errors import CompilationError


class ScheduleLegalizationError(CompilationError):
    """A circuit cannot be represented by the bounded logical scheduler."""


@dataclass(frozen=True)
class ScheduledInstruction:
    """One source instruction's deterministic dependency evidence."""

    instruction_index: int
    layer: int
    predecessors: tuple[int, ...]
    dependency_kinds: tuple[str, ...]


@dataclass(frozen=True)
class CircuitSchedule:
    """Immutable logical schedule without a second circuit representation."""

    program: CircuitIR
    target_snapshot_id: str
    layers: tuple[tuple[int, ...], ...]
    instructions: tuple[ScheduledInstruction, ...]
    depth: int
    maximum_parallel_width: int
    dynamic_dependency_count: int
    schedule_identity: str


def _condition_bits(instruction: Instruction, index: int) -> tuple[int, ...]:
    metadata = instruction.metadata
    if "conditions" in metadata and "condition_clauses" in metadata:
        raise ScheduleLegalizationError(
            f"instruction {index} defines both conditions and condition_clauses"
        )
    if "condition_clauses" in metadata:
        raw_clauses: Any = metadata["condition_clauses"]
    elif "conditions" in metadata:
        raw_clauses = (metadata["conditions"],)
    else:
        return ()

    try:
        clauses = tuple(tuple(clause) for clause in raw_clauses)
    except TypeError as error:
        raise ScheduleLegalizationError(
            f"instruction {index} classical conditions must be nested bit/value pairs"
        ) from error
    if not clauses or any(not clause for clause in clauses):
        raise ScheduleLegalizationError(
            f"instruction {index} classical conditions cannot be empty"
        )

    bits: set[int] = set()
    try:
        for clause in clauses:
            for pair in clause:
                bit, value = pair
                if type(bit) is not int or bit < 0:
                    raise ValueError("classical bit must be a non-negative integer")
                if type(value) is not int or value not in {0, 1}:
                    raise ValueError("classical condition value must be 0 or 1")
                bits.add(bit)
    except (TypeError, ValueError) as error:
        raise ScheduleLegalizationError(
            f"instruction {index} has malformed classical conditions: {error}"
        ) from error
    return tuple(sorted(bits))


def _measurement_bit(instruction: Instruction, index: int) -> int | None:
    if instruction.name != "measure":
        return None
    bit = instruction.metadata.get("classical_bit")
    if type(bit) is not int or bit < 0:
        raise ScheduleLegalizationError(
            f"measurement instruction {index} requires a non-negative integer "
            "classical_bit"
        )
    return bit


def _schedule_identity(
    program: CircuitIR,
    target_snapshot_id: str,
    instructions: Iterable[ScheduledInstruction],
) -> str:
    payload = {
        "circuit_content_hash": program.content_hash,
        "target_snapshot_id": target_snapshot_id,
        "instructions": [
            {
                "instruction_index": item.instruction_index,
                "layer": item.layer,
                "predecessors": item.predecessors,
                "dependency_kinds": item.dependency_kinds,
            }
            for item in instructions
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schedule_circuit_dependencies(
    program: object,
    *,
    target_snapshot_id: str,
    max_depth: int | None = None,
) -> CircuitSchedule:
    """Build a deterministic ASAP schedule with explicit causal dependencies.

    Layers are logical unit-time layers. Dynamic operations, conditions, and
    channels are conservative global barriers because target timing and
    concurrency evidence are not part of this contract.
    """

    ir = ensure_circuit_ir(program)
    snapshot_id = str(target_snapshot_id).strip()
    if not snapshot_id:
        raise ScheduleLegalizationError("target_snapshot_id cannot be empty")
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ScheduleLegalizationError("max_depth must be a non-negative integer")

    last_by_wire: dict[int, int] = {}
    classical_producers: dict[int, int] = {}
    last_barrier: int | None = None
    records: list[ScheduledInstruction] = []
    layer_by_instruction: dict[int, int] = {}
    dynamic_dependency_count = 0

    for index, instruction in enumerate(ir.instructions):
        condition_bits = _condition_bits(instruction, index)
        measurement_bit = _measurement_bit(instruction, index)
        is_barrier = bool(
            instruction.metadata.get("is_dynamic")
            or instruction.metadata.get("is_channel")
            or condition_bits
        )

        dependency_kinds: dict[int, set[str]] = {}

        def add_dependency(predecessor: int | None, kind: str) -> None:
            if predecessor is not None:
                dependency_kinds.setdefault(predecessor, set()).add(kind)

        for wire in instruction.wires:
            add_dependency(last_by_wire.get(wire), "wire")
        add_dependency(last_barrier, "barrier")

        for bit in condition_bits:
            if bit not in classical_producers:
                raise ScheduleLegalizationError(
                    f"instruction {index} reads classical bit {bit} before measurement"
                )
            add_dependency(classical_producers[bit], "classical")

        if is_barrier:
            for predecessor in set(last_by_wire.values()):
                add_dependency(predecessor, "barrier")

        predecessors = tuple(sorted(dependency_kinds))
        layer = 1 + max(
            (layer_by_instruction[predecessor] for predecessor in predecessors),
            default=-1,
        )
        kinds = tuple(
            sorted({kind for values in dependency_kinds.values() for kind in values})
        )
        if "classical" in kinds:
            dynamic_dependency_count += 1
        records.append(
            ScheduledInstruction(
                instruction_index=index,
                layer=layer,
                predecessors=predecessors,
                dependency_kinds=kinds,
            )
        )
        layer_by_instruction[index] = layer

        for wire in instruction.wires:
            last_by_wire[wire] = index
        if measurement_bit is not None:
            classical_producers[measurement_bit] = index
        if is_barrier:
            last_barrier = index
            for wire in range(ir.n_wires):
                last_by_wire[wire] = index

    depth = 1 + max(layer_by_instruction.values(), default=-1)
    if max_depth is not None and depth > max_depth:
        raise ScheduleLegalizationError(
            f"logical schedule depth {depth} exceeds maximum {max_depth}"
        )

    mutable_layers: list[list[int]] = [[] for _ in range(depth)]
    for record in records:
        mutable_layers[record.layer].append(record.instruction_index)
    layers = tuple(tuple(layer) for layer in mutable_layers)
    for layer_index, layer in enumerate(layers):
        occupied: set[int] = set()
        for instruction_index in layer:
            wires = set(ir.instructions[instruction_index].wires)
            if not occupied.isdisjoint(wires):
                raise ScheduleLegalizationError(
                    f"logical schedule layer {layer_index} repeats a wire"
                )
            occupied.update(wires)

    immutable_records = tuple(records)
    return CircuitSchedule(
        program=ir,
        target_snapshot_id=snapshot_id,
        layers=layers,
        instructions=immutable_records,
        depth=depth,
        maximum_parallel_width=max((len(layer) for layer in layers), default=0),
        dynamic_dependency_count=dynamic_dependency_count,
        schedule_identity=_schedule_identity(ir, snapshot_id, immutable_records),
    )


__all__ = (
    "CircuitSchedule",
    "ScheduleLegalizationError",
    "ScheduledInstruction",
    "schedule_circuit_dependencies",
)
