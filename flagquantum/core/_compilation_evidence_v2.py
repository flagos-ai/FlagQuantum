"""Version 2 compilation evidence for directed topology and explicit layout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._compilation_evidence import (
    _DEPENDENCY_KINDS,
    _MAX_BUNDLE_BYTES,
    _MAX_PREDECESSORS,
    _MAX_RECORDS,
    _OUTPUT_FIELDS,
    _SCHEMA,
    _SOURCE_FIELDS,
    _TARGET_FIELDS,
    _TOP_LEVEL_FIELDS,
    MappingTransitionEvidence,
    _bounded_string,
    _canonical_bytes,
    _closed,
    _index,
    _int_tuple,
    _sha256,
    _swap_layout,
    _wire_pair,
)

COMPILATION_EVIDENCE_VERSION_V2 = "2.0"
_COUPLING_FIELDS_V2 = {"n_wires", "directed_edges", "direction_semantics"}
_PLAN_FIELDS_V2 = {
    "plan_identity",
    "source_circuit_hash",
    "physical_circuit_hash",
    "topology_identity",
    "coupling",
    "initial_logical_to_physical",
    "pre_restore_logical_to_physical",
    "final_logical_to_physical",
    "mapping_transitions",
    "instructions",
    "topology_legalization_identity",
    "native_gate_legalization_identity",
    "direction_legalization_identity",
    "reversed_cx_count",
    "schedule_identity",
    "schedule_depth",
    "maximum_parallel_width",
    "critical_path",
}
_INSTRUCTION_FIELDS_V2 = {
    "instruction_index",
    "source_instruction_index",
    "topology_instruction_index",
    "native_replacement_ordinal",
    "native_instruction_index",
    "direction_replacement_ordinal",
    "direction_rewrite",
    "origin",
    "opcode",
    "logical_wires",
    "physical_wires",
    "layer",
    "predecessors",
    "dependency_kinds",
}
_ORIGINS_V2 = {
    "source",
    "topology_mapped",
    "native_decomposition",
    "routing_swap",
    "routing_swap_decomposition",
    "direction_rewrite",
    "routing_swap_direction_rewrite",
}
_DIRECTION_REWRITES = {"none", "reverse_cx_h_conjugation"}


@dataclass(frozen=True)
class DirectedCouplingEvidence:
    """Closed directed-CX coupling evidence."""

    n_wires: int
    directed_edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.n_wires) is not int or self.n_wires <= 0:
            raise ValueError("directed coupling n_wires must be positive")
        edges = tuple(
            _wire_pair(edge, "directed coupling edge", n_wires=self.n_wires)
            for edge in self.directed_edges
        )
        if edges != tuple(sorted(edges)) or len(edges) != len(set(edges)):
            raise ValueError("directed coupling edges must be sorted and unique")
        object.__setattr__(self, "directed_edges", edges)

    @property
    def topology_identity(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(
                {
                    "n_wires": self.n_wires,
                    "directed_edges": self.directed_edges,
                    "direction_semantics": "directed_cx",
                }
            )
        ).hexdigest()

    def has_edge(self, left: int, right: int) -> bool:
        return (left, right) in self.directed_edges

    def has_weak_edge(self, left: int, right: int) -> bool:
        return self.has_edge(left, right) or self.has_edge(right, left)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_wires": self.n_wires,
            "directed_edges": [list(edge) for edge in self.directed_edges],
            "direction_semantics": "directed_cx",
        }

    @classmethod
    def from_dict(cls, payload: object) -> DirectedCouplingEvidence:
        values = _closed(payload, _COUPLING_FIELDS_V2, "directed coupling evidence")
        if values["direction_semantics"] != "directed_cx":
            raise ValueError("coupling direction_semantics must be directed_cx")
        raw_edges = values["directed_edges"]
        if not isinstance(raw_edges, (tuple, list)):
            raise TypeError("directed coupling edges must be an array")
        return cls(n_wires=values["n_wires"], directed_edges=tuple(raw_edges))


@dataclass(frozen=True)
class PhysicalInstructionEvidenceV2:
    """One final instruction with native and direction-rewrite lineage."""

    instruction_index: int
    source_instruction_index: int
    topology_instruction_index: int
    native_replacement_ordinal: int
    native_instruction_index: int
    direction_replacement_ordinal: int
    direction_rewrite: str
    origin: str
    opcode: str
    logical_wires: tuple[int, ...]
    physical_wires: tuple[int, ...]
    layer: int
    predecessors: tuple[int, ...]
    dependency_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        for owner in (
            "instruction_index",
            "source_instruction_index",
            "topology_instruction_index",
            "native_replacement_ordinal",
            "native_instruction_index",
            "direction_replacement_ordinal",
            "layer",
        ):
            _index(getattr(self, owner), owner)
        if self.direction_rewrite not in _DIRECTION_REWRITES:
            raise ValueError("physical instruction has unsupported direction rewrite")
        if self.origin not in _ORIGINS_V2:
            raise ValueError("physical instruction has unsupported origin")
        _bounded_string(self.opcode, "physical instruction opcode")
        logical = _int_tuple(self.logical_wires, "logical_wires")
        physical = _int_tuple(self.physical_wires, "physical_wires")
        predecessors = _int_tuple(self.predecessors, "predecessors")
        if len(predecessors) > _MAX_PREDECESSORS:
            raise ValueError("physical instruction has too many predecessors")
        if predecessors != tuple(sorted(set(predecessors))):
            raise ValueError("predecessors must be sorted and unique")
        if any(item >= self.instruction_index for item in predecessors):
            raise ValueError("predecessors must precede the instruction")
        kinds = tuple(self.dependency_kinds)
        if any(
            type(item) is not str or item not in _DEPENDENCY_KINDS for item in kinds
        ) or kinds != tuple(sorted(set(kinds))):
            raise ValueError("dependency_kinds must be sorted supported values")
        object.__setattr__(self, "logical_wires", logical)
        object.__setattr__(self, "physical_wires", physical)
        object.__setattr__(self, "predecessors", predecessors)
        object.__setattr__(self, "dependency_kinds", kinds)

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction_index": self.instruction_index,
            "source_instruction_index": self.source_instruction_index,
            "topology_instruction_index": self.topology_instruction_index,
            "native_replacement_ordinal": self.native_replacement_ordinal,
            "native_instruction_index": self.native_instruction_index,
            "direction_replacement_ordinal": self.direction_replacement_ordinal,
            "direction_rewrite": self.direction_rewrite,
            "origin": self.origin,
            "opcode": self.opcode,
            "logical_wires": list(self.logical_wires),
            "physical_wires": list(self.physical_wires),
            "layer": self.layer,
            "predecessors": list(self.predecessors),
            "dependency_kinds": list(self.dependency_kinds),
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhysicalInstructionEvidenceV2:
        values = _closed(payload, _INSTRUCTION_FIELDS_V2, "physical instruction")
        return cls(**values)


def _critical_path(
    instructions: tuple[PhysicalInstructionEvidenceV2, ...], depth: int
) -> tuple[int, ...]:
    if not instructions:
        return ()
    current = min(
        (item for item in instructions if item.layer == depth - 1),
        key=lambda item: item.instruction_index,
    )
    path = [current.instruction_index]
    while current.predecessors:
        maximum_layer = max(instructions[index].layer for index in current.predecessors)
        predecessor = min(
            index
            for index in current.predecessors
            if instructions[index].layer == maximum_layer
        )
        path.append(predecessor)
        current = instructions[predecessor]
    return tuple(reversed(path))


def _validate_direction_groups(
    instructions: tuple[PhysicalInstructionEvidenceV2, ...],
) -> int:
    native_sequence = tuple(item.native_instruction_index for item in instructions)
    if native_sequence != tuple(sorted(native_sequence)):
        raise ValueError("native instruction groups must be contiguous and ordered")
    groups: dict[int, list[PhysicalInstructionEvidenceV2]] = {}
    for instruction in instructions:
        groups.setdefault(instruction.native_instruction_index, []).append(instruction)
    expected_native_indexes = list(range(len(groups)))
    if sorted(groups) != expected_native_indexes:
        raise ValueError("native instruction indexes must be dense and ordered")
    reversed_count = 0
    for native_index in expected_native_indexes:
        group = groups[native_index]
        rewrites = {item.direction_rewrite for item in group}
        if len(rewrites) != 1:
            raise ValueError("direction rewrite group is inconsistent")
        rewrite = group[0].direction_rewrite
        if rewrite == "none":
            if len(group) != 1 or group[0].direction_replacement_ordinal != 0:
                raise ValueError("unchanged direction group must contain one record")
            continue
        if (
            len(group) != 5
            or tuple(item.direction_replacement_ordinal for item in group)
            != tuple(range(5))
            or tuple(item.opcode for item in group) != ("h", "h", "cx", "h", "h")
        ):
            raise ValueError("reverse-CX direction group has invalid expansion")
        if (
            len(
                {
                    (
                        item.source_instruction_index,
                        item.topology_instruction_index,
                        item.native_replacement_ordinal,
                    )
                    for item in group
                }
            )
            != 1
        ):
            raise ValueError("reverse-CX direction group has inconsistent lineage")
        first, second, controlled, fourth, fifth = group
        if (
            first.physical_wires != fourth.physical_wires
            or second.physical_wires != fifth.physical_wires
            or controlled.physical_wires
            != (second.physical_wires[0], first.physical_wires[0])
        ):
            raise ValueError("reverse-CX direction group has invalid wire semantics")
        reversed_count += 1
    return reversed_count


@dataclass(frozen=True)
class PhysicalPlanEvidenceV2:
    """Strict directed-topology physical-plan evidence."""

    source_circuit_hash: str
    physical_circuit_hash: str
    target_snapshot_id: str
    topology_identity: str
    coupling: DirectedCouplingEvidence
    initial_logical_to_physical: tuple[int, ...]
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    mapping_transitions: tuple[MappingTransitionEvidence, ...]
    instructions: tuple[PhysicalInstructionEvidenceV2, ...]
    topology_legalization_identity: str
    native_gate_legalization_identity: str
    direction_legalization_identity: str
    reversed_cx_count: int
    schedule_identity: str
    schedule_depth: int
    maximum_parallel_width: int
    critical_path: tuple[int, ...]
    plan_identity: str = ""

    def __post_init__(self) -> None:
        for owner in (
            "source_circuit_hash",
            "physical_circuit_hash",
            "target_snapshot_id",
            "topology_identity",
            "topology_legalization_identity",
            "native_gate_legalization_identity",
            "direction_legalization_identity",
            "schedule_identity",
        ):
            _sha256(getattr(self, owner), owner)
        if not isinstance(self.coupling, DirectedCouplingEvidence):
            raise TypeError("coupling must be DirectedCouplingEvidence")
        if self.topology_identity != self.coupling.topology_identity:
            raise ValueError("directed coupling does not match topology identity")

        initial = _int_tuple(
            self.initial_logical_to_physical, "initial_logical_to_physical"
        )
        before_restore = _int_tuple(
            self.pre_restore_logical_to_physical,
            "pre_restore_logical_to_physical",
        )
        final = _int_tuple(self.final_logical_to_physical, "final_logical_to_physical")
        identity = tuple(range(len(initial)))
        if not initial or sorted(initial) != list(identity):
            raise ValueError("initial layout must be a dense permutation")
        if self.coupling.n_wires != len(initial):
            raise ValueError("directed physical and logical capacity must be equal")
        if sorted(before_restore) != list(identity):
            raise ValueError("pre-restore layout must be a dense permutation")
        if final != identity:
            raise ValueError("final logical-to-physical layout must be identity")

        transitions = tuple(self.mapping_transitions)
        instructions = tuple(self.instructions)
        if len(transitions) > _MAX_RECORDS:
            raise ValueError("physical plan has too many mapping transitions")
        if len(instructions) > _MAX_RECORDS:
            raise ValueError("physical plan has too many instruction records")
        if any(not isinstance(item, MappingTransitionEvidence) for item in transitions):
            raise TypeError("mapping transitions must be typed evidence records")
        if any(
            not isinstance(item, PhysicalInstructionEvidenceV2) for item in instructions
        ):
            raise TypeError("instructions must be typed version 2 evidence records")

        layout = list(initial)
        replay_pre_restore = tuple(layout)
        saw_restore = False
        for transition in transitions:
            if transition.layout_before != tuple(layout):
                raise ValueError("mapping transitions are not continuous")
            if transition.phase == "final_restore" and not saw_restore:
                replay_pre_restore = tuple(layout)
                saw_restore = True
            _swap_layout(layout, transition.physical_wires)
            if transition.layout_after != tuple(layout):
                raise ValueError("mapping transition result is inconsistent")
        if tuple(layout) != final:
            raise ValueError("mapping transitions do not reach final layout")
        if not saw_restore:
            replay_pre_restore = tuple(layout)
        if replay_pre_restore != before_restore:
            raise ValueError("pre-restore layout does not match transitions")

        for index, instruction in enumerate(instructions):
            if instruction.instruction_index != index:
                raise ValueError("physical instruction indexes must be dense")
            if any(
                instructions[item].layer >= instruction.layer
                for item in instruction.predecessors
            ):
                raise ValueError("physical dependencies must come from earlier layers")
            if bool(instruction.predecessors) != bool(instruction.dependency_kinds):
                raise ValueError("physical dependency kinds are inconsistent")
            if any(wire >= len(initial) for wire in instruction.logical_wires):
                raise ValueError("logical instruction wire is outside the layout")
            if any(
                wire >= self.coupling.n_wires for wire in instruction.physical_wires
            ):
                raise ValueError("physical instruction wire is outside the target")
            if len(instruction.physical_wires) == 2:
                left, right = instruction.physical_wires
                if instruction.opcode == "cx":
                    legal = self.coupling.has_edge(left, right)
                elif instruction.opcode == "swap":
                    legal = self.coupling.has_weak_edge(left, right)
                else:
                    legal = False
                if not legal:
                    raise ValueError("physical instruction violates directed coupling")

        reversed_count = _validate_direction_groups(instructions)
        if _index(self.reversed_cx_count, "reversed_cx_count") != reversed_count:
            raise ValueError("reversed_cx_count does not match direction lineage")
        depth = _index(self.schedule_depth, "schedule_depth")
        width = _index(self.maximum_parallel_width, "maximum_parallel_width")
        expected_depth = (
            0 if not instructions else 1 + max(i.layer for i in instructions)
        )
        expected_width = max(
            (
                sum(item.layer == layer for item in instructions)
                for layer in range(expected_depth)
            ),
            default=0,
        )
        if depth != expected_depth or width != expected_width:
            raise ValueError("physical plan schedule summary is inconsistent")
        critical_path = _int_tuple(self.critical_path, "critical_path")
        if critical_path != _critical_path(instructions, depth):
            raise ValueError("physical plan critical path is inconsistent")

        object.__setattr__(self, "initial_logical_to_physical", initial)
        object.__setattr__(self, "pre_restore_logical_to_physical", before_restore)
        object.__setattr__(self, "final_logical_to_physical", final)
        object.__setattr__(self, "mapping_transitions", transitions)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "critical_path", critical_path)
        expected_identity = hashlib.sha256(
            _canonical_bytes(self._plan_identity_payload())
        ).hexdigest()
        if self.plan_identity and self.plan_identity != expected_identity:
            raise ValueError("plan_identity does not match physical plan evidence")
        object.__setattr__(self, "plan_identity", expected_identity)

    def _plan_identity_payload(self) -> dict[str, object]:
        return {
            "schema": "flagquantum.physical_circuit_plan",
            "version": "2.0",
            "source_circuit_hash": self.source_circuit_hash,
            "physical_circuit_hash": self.physical_circuit_hash,
            "target_snapshot_id": self.target_snapshot_id,
            "topology_identity": self.topology_identity,
            "coupling_n_wires": self.coupling.n_wires,
            "coupling_edges": self.coupling.directed_edges,
            "coupling_direction_semantics": "directed_cx",
            "initial_logical_to_physical": self.initial_logical_to_physical,
            "pre_restore_logical_to_physical": self.pre_restore_logical_to_physical,
            "final_logical_to_physical": self.final_logical_to_physical,
            "mapping_transitions": [
                item.to_dict() for item in self.mapping_transitions
            ],
            "instructions": [item.to_dict() for item in self.instructions],
            "topology_legalization_identity": self.topology_legalization_identity,
            "native_gate_legalization_identity": self.native_gate_legalization_identity,
            "direction_legalization_identity": self.direction_legalization_identity,
            "reversed_cx_count": self.reversed_cx_count,
            "schedule_identity": self.schedule_identity,
            "schedule_depth": self.schedule_depth,
            "maximum_parallel_width": self.maximum_parallel_width,
            "critical_path": self.critical_path,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_identity": self.plan_identity,
            "source_circuit_hash": self.source_circuit_hash,
            "physical_circuit_hash": self.physical_circuit_hash,
            "topology_identity": self.topology_identity,
            "coupling": self.coupling.to_dict(),
            "initial_logical_to_physical": list(self.initial_logical_to_physical),
            "pre_restore_logical_to_physical": list(
                self.pre_restore_logical_to_physical
            ),
            "final_logical_to_physical": list(self.final_logical_to_physical),
            "mapping_transitions": [
                item.to_dict() for item in self.mapping_transitions
            ],
            "instructions": [item.to_dict() for item in self.instructions],
            "topology_legalization_identity": self.topology_legalization_identity,
            "native_gate_legalization_identity": self.native_gate_legalization_identity,
            "direction_legalization_identity": self.direction_legalization_identity,
            "reversed_cx_count": self.reversed_cx_count,
            "schedule_identity": self.schedule_identity,
            "schedule_depth": self.schedule_depth,
            "maximum_parallel_width": self.maximum_parallel_width,
            "critical_path": list(self.critical_path),
        }

    @classmethod
    def from_dict(
        cls, payload: object, *, target_snapshot_id: str
    ) -> PhysicalPlanEvidenceV2:
        values = _closed(payload, _PLAN_FIELDS_V2, "physical plan evidence v2")
        coupling = DirectedCouplingEvidence.from_dict(values.pop("coupling"))
        transitions = values.pop("mapping_transitions")
        instructions = values.pop("instructions")
        if not isinstance(transitions, (tuple, list)):
            raise TypeError("mapping_transitions must be an array")
        if not isinstance(instructions, (tuple, list)):
            raise TypeError("instructions must be an array")
        if len(transitions) > _MAX_RECORDS or len(instructions) > _MAX_RECORDS:
            raise ValueError("physical plan exceeds record limits")
        return cls(
            target_snapshot_id=target_snapshot_id,
            coupling=coupling,
            mapping_transitions=tuple(
                MappingTransitionEvidence.from_dict(item) for item in transitions
            ),
            instructions=tuple(
                PhysicalInstructionEvidenceV2.from_dict(item) for item in instructions
            ),
            **values,
        )


@dataclass(frozen=True)
class CompilationEvidenceBundleV2:
    """Strict version 2 compilation-evidence envelope."""

    producer: str
    source: Mapping[str, str | None]
    target: Mapping[str, str]
    physical_plan: PhysicalPlanEvidenceV2
    output: Mapping[str, str]
    bundle_identity: str = ""
    version: str = COMPILATION_EVIDENCE_VERSION_V2

    def __post_init__(self) -> None:
        if self.version != COMPILATION_EVIDENCE_VERSION_V2:
            raise ValueError(
                f"unsupported compilation evidence version {self.version!r}"
            )
        producer = _bounded_string(self.producer, "compilation evidence producer")
        source = _closed(self.source, _SOURCE_FIELDS, "compilation source")
        target = _closed(self.target, _TARGET_FIELDS, "compilation target")
        output = _closed(self.output, _OUTPUT_FIELDS, "compilation output")
        for name, value in source.items():
            _sha256(value, f"source.{name}", optional=name == "binding_identity")
        for name, value in target.items():
            _sha256(value, f"target.{name}")
        for name, value in output.items():
            if name == "profile":
                _bounded_string(value, "output.profile")
            else:
                _sha256(value, f"output.{name}")
        if not isinstance(self.physical_plan, PhysicalPlanEvidenceV2):
            raise TypeError("physical_plan must be PhysicalPlanEvidenceV2")
        if (
            source["source_circuit_hash"] != self.physical_plan.source_circuit_hash
            or source["final_circuit_hash"] != self.physical_plan.physical_circuit_hash
            or target["snapshot_id"] != self.physical_plan.target_snapshot_id
        ):
            raise ValueError("compilation evidence lineage is inconsistent")
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "source", MappingProxyType(dict(source)))
        object.__setattr__(self, "target", MappingProxyType(dict(target)))
        object.__setattr__(self, "output", MappingProxyType(dict(output)))
        expected = hashlib.sha256(
            _canonical_bytes(self._identity_payload())
        ).hexdigest()
        if self.bundle_identity and self.bundle_identity != expected:
            raise ValueError("bundle_identity does not match compilation evidence")
        object.__setattr__(self, "bundle_identity", expected)
        if len(_canonical_bytes(self.to_dict())) > _MAX_BUNDLE_BYTES:
            raise ValueError("compilation evidence exceeds maximum UTF-8 bytes")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "version": self.version,
            "producer": self.producer,
            "source": dict(self.source),
            "target": dict(self.target),
            "physical_plan": self.physical_plan.to_dict(),
            "output": dict(self.output),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "bundle_identity": self.bundle_identity}

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, payload: object) -> CompilationEvidenceBundleV2:
        values = _closed(payload, _TOP_LEVEL_FIELDS, "compilation evidence bundle")
        if values.pop("schema") != _SCHEMA:
            raise ValueError("invalid compilation evidence schema")
        target = _closed(values["target"], _TARGET_FIELDS, "compilation target")
        return cls(
            producer=values["producer"],
            version=values["version"],
            source=values["source"],
            target=target,
            physical_plan=PhysicalPlanEvidenceV2.from_dict(
                values["physical_plan"],
                target_snapshot_id=target["snapshot_id"],
            ),
            output=values["output"],
            bundle_identity=values["bundle_identity"],
        )


__all__ = (
    "COMPILATION_EVIDENCE_VERSION_V2",
    "CompilationEvidenceBundleV2",
    "DirectedCouplingEvidence",
    "PhysicalInstructionEvidenceV2",
    "PhysicalPlanEvidenceV2",
)
