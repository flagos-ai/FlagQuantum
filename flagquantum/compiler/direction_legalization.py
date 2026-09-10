"""Direction-correct native CX legalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.target_capabilities import TargetCapabilitySnapshot
from ..errors import CompilationError
from .directed_topology import DirectedCouplingMap


class DirectionLegalizationError(CompilationError):
    """A native circuit cannot satisfy an ordered coupling graph."""


@dataclass(frozen=True)
class DirectionLegalizationResult:
    """Direction-legal CircuitIR and deterministic rewrite evidence."""

    source_program: CircuitIR
    program: CircuitIR
    target_snapshot_id: str
    topology_identity: str
    reversed_cx_count: int
    legalization_identity: str

    def __post_init__(self) -> None:
        if self.reversed_cx_count < 0:
            raise ValueError("reversed_cx_count must be non-negative")
        payload = {
            "source_content_hash": self.source_program.content_hash,
            "result_content_hash": self.program.content_hash,
            "target_snapshot_id": self.target_snapshot_id,
            "topology_identity": self.topology_identity,
            "reversed_cx_count": self.reversed_cx_count,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if self.legalization_identity != expected:
            raise ValueError(
                "legalization_identity does not match directed-CX evidence"
            )


def legalize_directed_cx(
    program: object,
    *,
    coupling_map: DirectedCouplingMap,
    snapshot: TargetCapabilitySnapshot,
    native_opcodes: tuple[str, ...],
    max_added_operations: int = 256,
) -> DirectionLegalizationResult:
    """Preserve CX operand semantics on ordered hardware edges."""

    source = ensure_circuit_ir(program)
    if not isinstance(coupling_map, DirectedCouplingMap):
        raise TypeError("coupling_map must be a DirectedCouplingMap")
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a TargetCapabilitySnapshot")
    if type(max_added_operations) is not int or max_added_operations < 0:
        raise ValueError("max_added_operations must be a non-negative integer")
    native = frozenset(native_opcodes)
    output: list[Instruction] = []
    reversed_count = 0

    def emit(
        instruction: Instruction,
        *,
        native_index: int,
        ordinal: int,
        rewrite: str,
    ) -> None:
        output.append(
            Instruction(
                instruction.name,
                instruction.wires,
                params=instruction.params,
                matrix=instruction.matrix,
                metadata=dict(instruction.metadata)
                | {
                    "native_instruction_index": native_index,
                    "direction_replacement_ordinal": ordinal,
                    "direction_rewrite": rewrite,
                },
            )
        )

    for native_index, instruction in enumerate(source.instructions):
        if len(instruction.wires) != 2:
            emit(
                instruction,
                native_index=native_index,
                ordinal=0,
                rewrite="none",
            )
            continue
        if instruction.name == "swap":
            if not coupling_map.has_weak_edge(*instruction.wires):
                raise DirectionLegalizationError(
                    f"directed topology has no physical link for SWAP "
                    f"{instruction.wires[0]}<->{instruction.wires[1]}"
                )
            emit(
                instruction,
                native_index=native_index,
                ordinal=0,
                rewrite="none",
            )
            continue
        if instruction.name != "cx":
            raise DirectionLegalizationError(
                f"directed topology does not define semantics for two-wire "
                f"instruction {instruction.name!r}"
            )
        control, target = instruction.wires
        if coupling_map.has_edge(control, target):
            emit(
                instruction,
                native_index=native_index,
                ordinal=0,
                rewrite="none",
            )
            continue
        if not coupling_map.has_edge(target, control):
            raise DirectionLegalizationError(
                f"directed CX edge {control}->{target} is unavailable"
            )
        if not {"h", "cx"} <= native:
            raise DirectionLegalizationError(
                "reverse-CX legalization requires target-native h and cx gates"
            )
        rewrite = (
            Instruction("h", (control,)),
            Instruction("h", (target,)),
            Instruction("cx", (target, control)),
            Instruction("h", (control,)),
            Instruction("h", (target,)),
        )
        for ordinal, replacement in enumerate(rewrite):
            replacement = replace(replacement, metadata=dict(instruction.metadata))
            emit(
                replacement,
                native_index=native_index,
                ordinal=ordinal,
                rewrite="reverse_cx_h_conjugation",
            )
        reversed_count += 1

    if len(output) - len(source.instructions) > max_added_operations:
        raise DirectionLegalizationError(
            "directed-CX legalization exceeds max_added_operations"
        )
    result = replace(source, instructions=tuple(output))
    for index, instruction in enumerate(result.instructions):
        if instruction.name not in native:
            raise DirectionLegalizationError(
                f"direction-legal instruction {index} is not target-native"
            )
        if len(instruction.wires) == 2:
            if instruction.name == "cx" and not coupling_map.has_edge(
                *instruction.wires
            ):
                raise DirectionLegalizationError(
                    f"direction-legal CX instruction {index} violates ordered topology"
                )
            if instruction.name == "swap" and not coupling_map.has_weak_edge(
                *instruction.wires
            ):
                raise DirectionLegalizationError(
                    f"direction-legal SWAP instruction {index} violates topology"
                )
    payload = {
        "source_content_hash": source.content_hash,
        "result_content_hash": result.content_hash,
        "target_snapshot_id": snapshot.snapshot_id,
        "topology_identity": coupling_map.topology_identity,
        "reversed_cx_count": reversed_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return DirectionLegalizationResult(
        source_program=source,
        program=result,
        target_snapshot_id=snapshot.snapshot_id,
        topology_identity=coupling_map.topology_identity,
        reversed_cx_count=reversed_count,
        legalization_identity=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


__all__ = (
    "DirectionLegalizationError",
    "DirectionLegalizationResult",
    "legalize_directed_cx",
)
