"""Versioned, backend-neutral evidence for verified physical compilation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

COMPILATION_EVIDENCE_VERSION = "1.0"
_SCHEMA = "flagquantum.compilation_evidence_bundle"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_BUNDLE_BYTES = 16 * 1024 * 1024
_MAX_RECORDS = 4096
_MAX_PREDECESSORS = 4096
_MAX_STRING_BYTES = 256
_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "producer",
    "source",
    "target",
    "physical_plan",
    "output",
    "bundle_identity",
}
_SOURCE_FIELDS = {
    "source_artifact_identity",
    "circuit_artifact_identity",
    "binding_identity",
    "source_circuit_hash",
    "final_circuit_hash",
}
_TARGET_FIELDS = {"snapshot_id", "target_legalization_identity"}
_OUTPUT_FIELDS = {
    "profile",
    "payload_sha256",
    "emission_identity",
    "conformance_identity",
    "executable_artifact_identity",
    "artifact_compilation_identity",
}
_PLAN_FIELDS = {
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
    "schedule_identity",
    "schedule_depth",
    "maximum_parallel_width",
    "critical_path",
}
_COUPLING_FIELDS = {"n_wires", "edges", "direction_semantics"}
_TRANSITION_FIELDS = {
    "routed_instruction_index",
    "source_instruction_index",
    "phase",
    "physical_wires",
    "layout_before",
    "layout_after",
}
_INSTRUCTION_FIELDS = {
    "instruction_index",
    "source_instruction_index",
    "topology_instruction_index",
    "native_replacement_ordinal",
    "origin",
    "opcode",
    "logical_wires",
    "physical_wires",
    "layer",
    "predecessors",
    "dependency_kinds",
}
_ROUTING_PHASES = {"forward", "gate_restore", "final_restore"}
_ORIGINS = {
    "source",
    "topology_mapped",
    "native_decomposition",
    "routing_swap",
    "routing_swap_decomposition",
}
_DEPENDENCY_KINDS = {"wire", "barrier", "classical"}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _closed(value: object, fields: set[str], owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise ValueError(f"unknown {owner} field(s): " + ", ".join(sorted(unknown)))
    if missing:
        raise ValueError(f"missing {owner} field(s): " + ", ".join(sorted(missing)))
    return dict(value)


def _sha256(value: object, owner: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{owner} must be a lowercase SHA-256 digest")
    return value


def _bounded_string(value: object, owner: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{owner} must be a non-empty string")
    if len(value.encode("utf-8")) > _MAX_STRING_BYTES:
        raise ValueError(f"{owner} exceeds maximum UTF-8 bytes {_MAX_STRING_BYTES}")
    if value.startswith(("/", "~/", "file://", "http://", "https://")):
        raise ValueError(f"{owner} contains a prohibited locator")
    return value


def _index(value: object, owner: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{owner} must be a non-negative integer")
    return value


def _int_tuple(value: object, owner: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{owner} must be an array")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{owner} must contain non-negative integers")
    return result


def _wire_pair(value: object, owner: str, *, n_wires: int) -> tuple[int, int]:
    wires = _int_tuple(value, owner)
    if len(wires) != 2 or wires[0] == wires[1]:
        raise ValueError(f"{owner} must contain two distinct wires")
    if any(wire >= n_wires for wire in wires):
        raise ValueError(f"{owner} contains a wire outside the coupling map")
    return wires


@dataclass(frozen=True)
class CouplingEvidence:
    """Closed undirected coupling-map evidence."""

    n_wires: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.n_wires) is not int or self.n_wires <= 0:
            raise ValueError("coupling n_wires must be a positive integer")
        edges = tuple(
            _wire_pair(edge, "coupling edge", n_wires=self.n_wires)
            for edge in self.edges
        )
        if any(left > right for left, right in edges):
            raise ValueError("coupling edges must use normalized wire order")
        if len(set(edges)) != len(edges):
            raise ValueError("coupling edges must be unique")
        object.__setattr__(self, "edges", edges)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_wires": self.n_wires,
            "edges": [list(edge) for edge in self.edges],
            "direction_semantics": "undirected",
        }

    @classmethod
    def from_dict(cls, payload: object) -> CouplingEvidence:
        values = _closed(payload, _COUPLING_FIELDS, "coupling evidence")
        if values["direction_semantics"] != "undirected":
            raise ValueError("coupling direction_semantics must be undirected")
        raw_edges = values["edges"]
        if not isinstance(raw_edges, (tuple, list)):
            raise TypeError("coupling edges must be an array")
        return cls(n_wires=values["n_wires"], edges=tuple(raw_edges))


@dataclass(frozen=True)
class MappingTransitionEvidence:
    """One serialized routing layout transition."""

    routed_instruction_index: int
    source_instruction_index: int
    phase: str
    physical_wires: tuple[int, int]
    layout_before: tuple[int, ...]
    layout_after: tuple[int, ...]

    def __post_init__(self) -> None:
        _index(self.routed_instruction_index, "routed_instruction_index")
        _index(self.source_instruction_index, "source_instruction_index")
        if self.phase not in _ROUTING_PHASES:
            raise ValueError("mapping transition has unsupported routing phase")
        before = _int_tuple(self.layout_before, "layout_before")
        after = _int_tuple(self.layout_after, "layout_after")
        if not before or sorted(before) != list(range(len(before))):
            raise ValueError("layout_before must be a dense permutation")
        if sorted(after) != list(range(len(before))):
            raise ValueError("layout_after must be a matching dense permutation")
        wires = _wire_pair(
            self.physical_wires,
            "mapping transition physical_wires",
            n_wires=len(before),
        )
        object.__setattr__(self, "physical_wires", wires)
        object.__setattr__(self, "layout_before", before)
        object.__setattr__(self, "layout_after", after)

    def to_dict(self) -> dict[str, object]:
        return {
            "routed_instruction_index": self.routed_instruction_index,
            "source_instruction_index": self.source_instruction_index,
            "phase": self.phase,
            "physical_wires": list(self.physical_wires),
            "layout_before": list(self.layout_before),
            "layout_after": list(self.layout_after),
        }

    @classmethod
    def from_dict(cls, payload: object) -> MappingTransitionEvidence:
        values = _closed(payload, _TRANSITION_FIELDS, "mapping transition")
        return cls(**values)


@dataclass(frozen=True)
class PhysicalInstructionEvidence:
    """One serialized final instruction provenance and dependency record."""

    instruction_index: int
    source_instruction_index: int
    topology_instruction_index: int
    native_replacement_ordinal: int
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
            "layer",
        ):
            _index(getattr(self, owner), owner)
        if self.origin not in _ORIGINS:
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
            "origin": self.origin,
            "opcode": self.opcode,
            "logical_wires": list(self.logical_wires),
            "physical_wires": list(self.physical_wires),
            "layer": self.layer,
            "predecessors": list(self.predecessors),
            "dependency_kinds": list(self.dependency_kinds),
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhysicalInstructionEvidence:
        values = _closed(payload, _INSTRUCTION_FIELDS, "physical instruction")
        return cls(**values)


def _swap_layout(layout: list[int], wires: tuple[int, int]) -> None:
    inverse = {physical: logical for logical, physical in enumerate(layout)}
    left, right = wires
    layout[inverse[left]], layout[inverse[right]] = right, left


def _coupling_identity(coupling: CouplingEvidence) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "n_wires": coupling.n_wires,
                "edges": coupling.edges,
                "direction_semantics": "undirected",
            }
        )
    ).hexdigest()


def _critical_path(
    instructions: tuple[PhysicalInstructionEvidence, ...], depth: int
) -> tuple[int, ...]:
    if not instructions:
        return ()
    current = min(
        (item for item in instructions if item.layer == depth - 1),
        key=lambda item: item.instruction_index,
    )
    reversed_path = [current.instruction_index]
    while current.predecessors:
        maximum_layer = max(instructions[index].layer for index in current.predecessors)
        predecessor = min(
            index
            for index in current.predecessors
            if instructions[index].layer == maximum_layer
        )
        reversed_path.append(predecessor)
        current = instructions[predecessor]
    return tuple(reversed(reversed_path))


@dataclass(frozen=True)
class PhysicalPlanEvidence:
    """Serializable projection of a verified Compiler physical plan."""

    source_circuit_hash: str
    physical_circuit_hash: str
    target_snapshot_id: str
    topology_identity: str | None
    coupling: CouplingEvidence | None
    initial_logical_to_physical: tuple[int, ...]
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    mapping_transitions: tuple[MappingTransitionEvidence, ...]
    instructions: tuple[PhysicalInstructionEvidence, ...]
    topology_legalization_identity: str | None
    native_gate_legalization_identity: str
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
            "native_gate_legalization_identity",
            "schedule_identity",
        ):
            _sha256(getattr(self, owner), owner)
        topology_identity = _sha256(
            self.topology_identity, "topology_identity", optional=True
        )
        topology_legalization_identity = _sha256(
            self.topology_legalization_identity,
            "topology_legalization_identity",
            optional=True,
        )
        if (topology_identity is None) != (self.coupling is None):
            raise ValueError("topology identity and coupling evidence must coexist")
        if (topology_identity is None) != (topology_legalization_identity is None):
            raise ValueError("topology legalization evidence is inconsistent")
        if self.coupling is not None and topology_identity != _coupling_identity(
            self.coupling
        ):
            raise ValueError("coupling evidence does not match topology identity")

        initial = _int_tuple(
            self.initial_logical_to_physical, "initial_logical_to_physical"
        )
        before_restore = _int_tuple(
            self.pre_restore_logical_to_physical,
            "pre_restore_logical_to_physical",
        )
        final = _int_tuple(self.final_logical_to_physical, "final_logical_to_physical")
        if not initial or initial != tuple(range(len(initial))):
            raise ValueError("initial logical-to-physical layout must be identity")
        if sorted(before_restore) != list(range(len(initial))):
            raise ValueError("pre-restore layout must be a dense permutation")
        if final != initial:
            raise ValueError("final logical-to-physical layout must be identity")
        if self.coupling is not None and self.coupling.n_wires < len(initial):
            raise ValueError("coupling map has fewer wires than the logical layout")

        transitions = tuple(self.mapping_transitions)
        instructions = tuple(self.instructions)
        if len(transitions) > _MAX_RECORDS:
            raise ValueError("physical plan has too many mapping transitions")
        if len(instructions) > _MAX_RECORDS:
            raise ValueError("physical plan has too many instruction records")
        if any(not isinstance(item, MappingTransitionEvidence) for item in transitions):
            raise TypeError("mapping transitions must be typed evidence records")
        if any(
            not isinstance(item, PhysicalInstructionEvidence) for item in instructions
        ):
            raise TypeError("instructions must be typed evidence records")
        if self.coupling is None and transitions:
            raise ValueError("mapping transitions require coupling evidence")

        layout = list(initial)
        replay_pre_restore = tuple(layout)
        saw_final_restore = False
        for transition in transitions:
            if transition.layout_before != tuple(layout):
                raise ValueError("mapping transitions are not continuous")
            if transition.phase == "final_restore" and not saw_final_restore:
                replay_pre_restore = tuple(layout)
                saw_final_restore = True
            _swap_layout(layout, transition.physical_wires)
            if transition.layout_after != tuple(layout):
                raise ValueError("mapping transition result is inconsistent")
        if tuple(layout) != final:
            raise ValueError("mapping transitions do not reach final layout")
        if not saw_final_restore:
            replay_pre_restore = tuple(layout)
        if replay_pre_restore != before_restore:
            raise ValueError("pre-restore layout does not match mapping transitions")

        for index, instruction in enumerate(instructions):
            if instruction.instruction_index != index:
                raise ValueError(
                    "physical instruction indexes must be dense and ordered"
                )
            if any(
                instructions[predecessor].layer >= instruction.layer
                for predecessor in instruction.predecessors
            ):
                raise ValueError("physical dependencies must come from earlier layers")
            if bool(instruction.predecessors) != bool(instruction.dependency_kinds):
                raise ValueError("physical dependency kinds are inconsistent")
            if any(wire >= len(initial) for wire in instruction.logical_wires):
                raise ValueError("logical instruction wire is outside the layout")
            physical_limit = (
                len(initial) if self.coupling is None else self.coupling.n_wires
            )
            if any(wire >= physical_limit for wire in instruction.physical_wires):
                raise ValueError("physical instruction wire is outside the target")
            if self.coupling is not None and len(instruction.physical_wires) == 2:
                edge = tuple(sorted(instruction.physical_wires))
                if edge not in self.coupling.edges:
                    raise ValueError("physical instruction violates coupling evidence")

        depth = _index(self.schedule_depth, "schedule_depth")
        width = _index(self.maximum_parallel_width, "maximum_parallel_width")
        expected_depth = (
            0 if not instructions else 1 + max(i.layer for i in instructions)
        )
        layer_widths = {
            layer: sum(item.layer == layer for item in instructions)
            for layer in range(expected_depth)
        }
        expected_width = max(layer_widths.values(), default=0)
        if depth != expected_depth or width != expected_width:
            raise ValueError("physical plan schedule summary is inconsistent")
        critical_path = _int_tuple(self.critical_path, "critical_path")
        if critical_path != _critical_path(instructions, depth):
            raise ValueError("physical plan critical path is inconsistent")

        object.__setattr__(self, "topology_identity", topology_identity)
        object.__setattr__(
            self, "topology_legalization_identity", topology_legalization_identity
        )
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
            "version": "1.0",
            "source_circuit_hash": self.source_circuit_hash,
            "physical_circuit_hash": self.physical_circuit_hash,
            "target_snapshot_id": self.target_snapshot_id,
            "topology_identity": self.topology_identity,
            "coupling_n_wires": (
                None if self.coupling is None else self.coupling.n_wires
            ),
            "coupling_edges": () if self.coupling is None else self.coupling.edges,
            "initial_logical_to_physical": self.initial_logical_to_physical,
            "pre_restore_logical_to_physical": self.pre_restore_logical_to_physical,
            "final_logical_to_physical": self.final_logical_to_physical,
            "mapping_transitions": [
                item.to_dict() for item in self.mapping_transitions
            ],
            "instructions": [item.to_dict() for item in self.instructions],
            "topology_legalization_identity": self.topology_legalization_identity,
            "native_gate_legalization_identity": self.native_gate_legalization_identity,
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
            "coupling": None if self.coupling is None else self.coupling.to_dict(),
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
            "schedule_identity": self.schedule_identity,
            "schedule_depth": self.schedule_depth,
            "maximum_parallel_width": self.maximum_parallel_width,
            "critical_path": list(self.critical_path),
        }

    @classmethod
    def from_dict(
        cls, payload: object, *, target_snapshot_id: str
    ) -> PhysicalPlanEvidence:
        values = _closed(payload, _PLAN_FIELDS, "physical plan evidence")
        coupling_value = values.pop("coupling")
        transitions = values.pop("mapping_transitions")
        instructions = values.pop("instructions")
        if not isinstance(transitions, (tuple, list)):
            raise TypeError("mapping_transitions must be an array")
        if not isinstance(instructions, (tuple, list)):
            raise TypeError("instructions must be an array")
        if len(transitions) > _MAX_RECORDS:
            raise ValueError("physical plan has too many mapping transitions")
        if len(instructions) > _MAX_RECORDS:
            raise ValueError("physical plan has too many instruction records")
        return cls(
            target_snapshot_id=target_snapshot_id,
            coupling=(
                None
                if coupling_value is None
                else CouplingEvidence.from_dict(coupling_value)
            ),
            mapping_transitions=tuple(
                MappingTransitionEvidence.from_dict(item) for item in transitions
            ),
            instructions=tuple(
                PhysicalInstructionEvidence.from_dict(item) for item in instructions
            ),
            **values,
        )


@dataclass(frozen=True)
class CompilationEvidenceBundle:
    """Strict serialized evidence joining compilation inputs, plan, and output."""

    producer: str
    source: Mapping[str, str | None]
    target: Mapping[str, str]
    physical_plan: PhysicalPlanEvidence
    output: Mapping[str, str]
    bundle_identity: str = ""
    version: str = COMPILATION_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != COMPILATION_EVIDENCE_VERSION:
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
        if not isinstance(self.physical_plan, PhysicalPlanEvidence):
            raise TypeError("physical_plan must be PhysicalPlanEvidence")
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
        expected_identity = hashlib.sha256(
            _canonical_bytes(self._identity_payload())
        ).hexdigest()
        if self.bundle_identity and self.bundle_identity != expected_identity:
            raise ValueError("bundle_identity does not match compilation evidence")
        object.__setattr__(self, "bundle_identity", expected_identity)
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
    def from_dict(cls, payload: object) -> CompilationEvidenceBundle:
        values = _closed(payload, _TOP_LEVEL_FIELDS, "compilation evidence bundle")
        if values.pop("schema") != _SCHEMA:
            raise ValueError("invalid compilation evidence schema")
        target = _closed(values["target"], _TARGET_FIELDS, "compilation target")
        return cls(
            producer=values["producer"],
            version=values["version"],
            source=values["source"],
            target=target,
            physical_plan=PhysicalPlanEvidence.from_dict(
                values["physical_plan"],
                target_snapshot_id=target["snapshot_id"],
            ),
            output=values["output"],
            bundle_identity=values["bundle_identity"],
        )


def read_compilation_evidence_bundle_json(payload: str) -> CompilationEvidenceBundle:
    """Decode a bundle while rejecting duplicate JSON keys at every depth."""

    if type(payload) is not str:
        raise TypeError("compilation evidence JSON must be a string")
    if len(payload.encode("utf-8")) > _MAX_BUNDLE_BYTES:
        raise ValueError("compilation evidence exceeds maximum UTF-8 bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate compilation evidence field {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError("compilation evidence must be valid JSON") from error
    return CompilationEvidenceBundle.from_dict(decoded)


__all__ = (
    "COMPILATION_EVIDENCE_VERSION",
    "CompilationEvidenceBundle",
    "CouplingEvidence",
    "MappingTransitionEvidence",
    "PhysicalInstructionEvidence",
    "PhysicalPlanEvidence",
    "read_compilation_evidence_bundle_json",
)
