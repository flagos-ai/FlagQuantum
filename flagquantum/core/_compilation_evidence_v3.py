"""Version 3 compilation evidence for allocated physical resources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._compilation_evidence import (
    _MAX_BUNDLE_BYTES,
    _MAX_RECORDS,
    _OUTPUT_FIELDS,
    _ROUTING_PHASES,
    _SCHEMA,
    _SOURCE_FIELDS,
    _TARGET_FIELDS,
    _TOP_LEVEL_FIELDS,
    _bounded_string,
    _canonical_bytes,
    _closed,
    _index,
    _int_tuple,
    _sha256,
    _wire_pair,
)
from ._compilation_evidence_v2 import (
    DirectedCouplingEvidence,
    PhysicalInstructionEvidenceV2,
    _critical_path,
    _validate_direction_groups,
)

COMPILATION_EVIDENCE_VERSION_V3 = "3.0"
_TRANSITION_FIELDS_V3 = {
    "routed_instruction_index",
    "source_instruction_index",
    "phase",
    "physical_wires",
    "layout_before",
    "layout_after",
    "physical_to_logical_before",
    "physical_to_logical_after",
}
_PLAN_FIELDS_V3 = {
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
    "logical_wire_count",
    "physical_slot_count",
    "initial_physical_to_logical",
    "pre_restore_physical_to_logical",
    "final_physical_to_logical",
    "logical_result_physical_slots",
    "allocation_identity",
}


def _layout(
    value: object, owner: str, *, logical: int, physical: int
) -> tuple[int, ...]:
    result = _int_tuple(value, owner)
    if len(result) != logical or len(set(result)) != logical:
        raise ValueError(f"{owner} must inject every logical wire exactly once")
    if any(item >= physical for item in result):
        raise ValueError(f"{owner} contains a slot outside physical capacity")
    return result


def _occupancy(
    value: object, owner: str, *, logical: int, physical: int
) -> tuple[int | None, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{owner} must be an array")
    result = tuple(value)
    if len(result) != physical:
        raise ValueError(f"{owner} must cover every physical slot")
    populated = tuple(item for item in result if item is not None)
    if any(type(item) is not int or item < 0 or item >= logical for item in populated):
        raise ValueError(f"{owner} contains an invalid logical wire")
    if sorted(populated) != list(range(logical)):
        raise ValueError(f"{owner} must contain every logical wire exactly once")
    return result


def _layout_from_occupancy(
    occupancy: tuple[int | None, ...], logical: int
) -> tuple[int, ...]:
    result = [-1] * logical
    for physical, logical_wire in enumerate(occupancy):
        if logical_wire is not None:
            result[logical_wire] = physical
    return tuple(result)


@dataclass(frozen=True)
class MappingTransitionEvidenceV3:
    """One allocated routing transition with nullable occupancy evidence."""

    routed_instruction_index: int
    source_instruction_index: int
    phase: str
    physical_wires: tuple[int, int]
    layout_before: tuple[int, ...]
    layout_after: tuple[int, ...]
    physical_to_logical_before: tuple[int | None, ...]
    physical_to_logical_after: tuple[int | None, ...]

    def __post_init__(self) -> None:
        _index(self.routed_instruction_index, "routed_instruction_index")
        _index(self.source_instruction_index, "source_instruction_index")
        if self.phase not in _ROUTING_PHASES:
            raise ValueError("mapping transition has unsupported routing phase")
        physical = len(self.physical_to_logical_before)
        logical = len(self.layout_before)
        if physical <= logical:
            raise ValueError("allocated transition requires spare physical capacity")
        before = _layout(
            self.layout_before, "layout_before", logical=logical, physical=physical
        )
        after = _layout(
            self.layout_after, "layout_after", logical=logical, physical=physical
        )
        occupancy_before = _occupancy(
            self.physical_to_logical_before,
            "physical_to_logical_before",
            logical=logical,
            physical=physical,
        )
        occupancy_after = _occupancy(
            self.physical_to_logical_after,
            "physical_to_logical_after",
            logical=logical,
            physical=physical,
        )
        if _layout_from_occupancy(occupancy_before, logical) != before:
            raise ValueError("mapping transition layout and occupancy before disagree")
        wires = _wire_pair(
            self.physical_wires,
            "mapping transition physical_wires",
            n_wires=physical,
        )
        expected = list(occupancy_before)
        left, right = wires
        expected[left], expected[right] = expected[right], expected[left]
        if tuple(expected) != occupancy_after:
            raise ValueError("mapping transition occupancy does not apply one SWAP")
        if _layout_from_occupancy(occupancy_after, logical) != after:
            raise ValueError("mapping transition layout and occupancy after disagree")
        object.__setattr__(self, "physical_wires", wires)
        object.__setattr__(self, "layout_before", before)
        object.__setattr__(self, "layout_after", after)
        object.__setattr__(self, "physical_to_logical_before", occupancy_before)
        object.__setattr__(self, "physical_to_logical_after", occupancy_after)

    def to_dict(self) -> dict[str, object]:
        return {
            "routed_instruction_index": self.routed_instruction_index,
            "source_instruction_index": self.source_instruction_index,
            "phase": self.phase,
            "physical_wires": list(self.physical_wires),
            "layout_before": list(self.layout_before),
            "layout_after": list(self.layout_after),
            "physical_to_logical_before": list(self.physical_to_logical_before),
            "physical_to_logical_after": list(self.physical_to_logical_after),
        }

    @classmethod
    def from_dict(cls, payload: object) -> MappingTransitionEvidenceV3:
        return cls(**_closed(payload, _TRANSITION_FIELDS_V3, "mapping transition v3"))


@dataclass(frozen=True)
class PhysicalPlanEvidenceV3:
    """Strict directed-topology plan evidence with allocated idle slots."""

    source_circuit_hash: str
    physical_circuit_hash: str
    target_snapshot_id: str
    topology_identity: str
    coupling: DirectedCouplingEvidence
    initial_logical_to_physical: tuple[int, ...]
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    mapping_transitions: tuple[MappingTransitionEvidenceV3, ...]
    instructions: tuple[PhysicalInstructionEvidenceV2, ...]
    topology_legalization_identity: str
    native_gate_legalization_identity: str
    direction_legalization_identity: str
    reversed_cx_count: int
    schedule_identity: str
    schedule_depth: int
    maximum_parallel_width: int
    critical_path: tuple[int, ...]
    logical_wire_count: int
    physical_slot_count: int
    initial_physical_to_logical: tuple[int | None, ...]
    pre_restore_physical_to_logical: tuple[int | None, ...]
    final_physical_to_logical: tuple[int | None, ...]
    logical_result_physical_slots: tuple[int, ...]
    allocation_identity: str
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
            "allocation_identity",
        ):
            _sha256(getattr(self, owner), owner)
        logical = _index(self.logical_wire_count, "logical_wire_count")
        physical = _index(self.physical_slot_count, "physical_slot_count")
        if logical <= 0 or physical <= logical:
            raise ValueError("version 3 plan requires spare physical capacity")
        if not isinstance(self.coupling, DirectedCouplingEvidence):
            raise TypeError("coupling must be DirectedCouplingEvidence")
        if self.coupling.n_wires != physical:
            raise ValueError("directed coupling must cover every physical slot")
        if self.topology_identity != self.coupling.topology_identity:
            raise ValueError("directed coupling does not match topology identity")

        initial = _layout(
            self.initial_logical_to_physical,
            "initial_logical_to_physical",
            logical=logical,
            physical=physical,
        )
        pre_restore = _layout(
            self.pre_restore_logical_to_physical,
            "pre_restore_logical_to_physical",
            logical=logical,
            physical=physical,
        )
        final = _layout(
            self.final_logical_to_physical,
            "final_logical_to_physical",
            logical=logical,
            physical=physical,
        )
        initial_occ = _occupancy(
            self.initial_physical_to_logical,
            "initial_physical_to_logical",
            logical=logical,
            physical=physical,
        )
        pre_restore_occ = _occupancy(
            self.pre_restore_physical_to_logical,
            "pre_restore_physical_to_logical",
            logical=logical,
            physical=physical,
        )
        final_occ = _occupancy(
            self.final_physical_to_logical,
            "final_physical_to_logical",
            logical=logical,
            physical=physical,
        )
        if (
            _layout_from_occupancy(initial_occ, logical) != initial
            or _layout_from_occupancy(pre_restore_occ, logical) != pre_restore
            or _layout_from_occupancy(final_occ, logical) != final
        ):
            raise ValueError("physical allocation layout and occupancy disagree")
        result_slots = _layout(
            self.logical_result_physical_slots,
            "logical_result_physical_slots",
            logical=logical,
            physical=physical,
        )
        if result_slots != final:
            raise ValueError("logical result projection does not match final layout")

        expected_allocation = hashlib.sha256(
            _canonical_bytes(
                {
                    "logical_wire_count": logical,
                    "physical_slot_count": physical,
                    "initial_logical_to_physical": initial,
                    "initial_physical_to_logical": initial_occ,
                    "logical_result_physical_slots": result_slots,
                    "workspace_initial_state": "standard_zero",
                    "workspace_cleanup": "inverse_routing_swaps",
                }
            )
        ).hexdigest()
        if self.allocation_identity != expected_allocation:
            raise ValueError("allocation_identity does not match allocation evidence")

        transitions = tuple(self.mapping_transitions)
        instructions = tuple(self.instructions)
        if len(transitions) > _MAX_RECORDS or len(instructions) > _MAX_RECORDS:
            raise ValueError("physical plan exceeds record limits")
        if any(
            not isinstance(item, MappingTransitionEvidenceV3) for item in transitions
        ):
            raise TypeError("mapping transitions must be typed version 3 evidence")
        if any(
            not isinstance(item, PhysicalInstructionEvidenceV2) for item in instructions
        ):
            raise TypeError("instructions must be typed version 2 evidence records")
        layout_state, occupancy_state = initial, initial_occ
        replay_pre_layout, replay_pre_occ = initial, initial_occ
        saw_restore = False
        for transition in transitions:
            if (
                transition.layout_before != layout_state
                or transition.physical_to_logical_before != occupancy_state
            ):
                raise ValueError("allocated mapping transitions are not continuous")
            if transition.phase == "final_restore" and not saw_restore:
                replay_pre_layout, replay_pre_occ = layout_state, occupancy_state
                saw_restore = True
            layout_state = transition.layout_after
            occupancy_state = transition.physical_to_logical_after
        if layout_state != final or occupancy_state != final_occ:
            raise ValueError("allocated mapping transitions do not reach final state")
        if not saw_restore:
            replay_pre_layout, replay_pre_occ = layout_state, occupancy_state
        if replay_pre_layout != pre_restore or replay_pre_occ != pre_restore_occ:
            raise ValueError("pre-restore allocation does not match transitions")

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
            if any(wire >= logical for wire in instruction.logical_wires):
                raise ValueError("logical instruction wire is outside logical capacity")
            if any(wire >= physical for wire in instruction.physical_wires):
                raise ValueError(
                    "physical instruction wire is outside physical capacity"
                )
            if len(instruction.physical_wires) == 2:
                left, right = instruction.physical_wires
                legal = (
                    self.coupling.has_edge(left, right)
                    if instruction.opcode == "cx"
                    else (
                        self.coupling.has_weak_edge(left, right)
                        if instruction.opcode == "swap"
                        else False
                    )
                )
                if not legal:
                    raise ValueError("physical instruction violates directed coupling")
        reversed_count = _validate_direction_groups(instructions)
        if _index(self.reversed_cx_count, "reversed_cx_count") != reversed_count:
            raise ValueError("reversed_cx_count does not match direction lineage")
        depth = _index(self.schedule_depth, "schedule_depth")
        width = _index(self.maximum_parallel_width, "maximum_parallel_width")
        expected_depth = (
            0 if not instructions else 1 + max(item.layer for item in instructions)
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

        for name, value in (
            ("initial_logical_to_physical", initial),
            ("pre_restore_logical_to_physical", pre_restore),
            ("final_logical_to_physical", final),
            ("initial_physical_to_logical", initial_occ),
            ("pre_restore_physical_to_logical", pre_restore_occ),
            ("final_physical_to_logical", final_occ),
            ("logical_result_physical_slots", result_slots),
            ("mapping_transitions", transitions),
            ("instructions", instructions),
            ("critical_path", critical_path),
        ):
            object.__setattr__(self, name, value)
        expected_identity = hashlib.sha256(
            _canonical_bytes(self._plan_identity_payload())
        ).hexdigest()
        if self.plan_identity and self.plan_identity != expected_identity:
            raise ValueError("plan_identity does not match physical plan evidence")
        object.__setattr__(self, "plan_identity", expected_identity)

    def _plan_identity_payload(self) -> dict[str, object]:
        return {
            "schema": "flagquantum.physical_circuit_plan",
            "version": "3.0",
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
            "logical_wire_count": self.logical_wire_count,
            "physical_slot_count": self.physical_slot_count,
            "initial_physical_to_logical": self.initial_physical_to_logical,
            "pre_restore_physical_to_logical": self.pre_restore_physical_to_logical,
            "final_physical_to_logical": self.final_physical_to_logical,
            "logical_result_physical_slots": self.logical_result_physical_slots,
            "allocation_identity": self.allocation_identity,
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
            "logical_wire_count": self.logical_wire_count,
            "physical_slot_count": self.physical_slot_count,
            "initial_physical_to_logical": list(self.initial_physical_to_logical),
            "pre_restore_physical_to_logical": list(
                self.pre_restore_physical_to_logical
            ),
            "final_physical_to_logical": list(self.final_physical_to_logical),
            "logical_result_physical_slots": list(self.logical_result_physical_slots),
            "allocation_identity": self.allocation_identity,
        }

    @classmethod
    def from_dict(
        cls, payload: object, *, target_snapshot_id: str
    ) -> PhysicalPlanEvidenceV3:
        values = _closed(payload, _PLAN_FIELDS_V3, "physical plan evidence v3")
        coupling = DirectedCouplingEvidence.from_dict(values.pop("coupling"))
        transitions, instructions = values.pop("mapping_transitions"), values.pop(
            "instructions"
        )
        if not isinstance(transitions, (tuple, list)) or not isinstance(
            instructions, (tuple, list)
        ):
            raise TypeError("mapping_transitions and instructions must be arrays")
        if len(transitions) > _MAX_RECORDS or len(instructions) > _MAX_RECORDS:
            raise ValueError("physical plan exceeds record limits")
        return cls(
            target_snapshot_id=target_snapshot_id,
            coupling=coupling,
            mapping_transitions=tuple(
                MappingTransitionEvidenceV3.from_dict(item) for item in transitions
            ),
            instructions=tuple(
                PhysicalInstructionEvidenceV2.from_dict(item) for item in instructions
            ),
            **values,
        )


@dataclass(frozen=True)
class CompilationEvidenceBundleV3:
    """Strict version 3 compilation-evidence envelope."""

    producer: str
    source: Mapping[str, str | None]
    target: Mapping[str, str]
    physical_plan: PhysicalPlanEvidenceV3
    output: Mapping[str, str]
    bundle_identity: str = ""
    version: str = COMPILATION_EVIDENCE_VERSION_V3

    def __post_init__(self) -> None:
        if self.version != COMPILATION_EVIDENCE_VERSION_V3:
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
            (
                _bounded_string(value, "output.profile")
                if name == "profile"
                else _sha256(value, f"output.{name}")
            )
        if not isinstance(self.physical_plan, PhysicalPlanEvidenceV3):
            raise TypeError("physical_plan must be PhysicalPlanEvidenceV3")
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
    def from_dict(cls, payload: object) -> CompilationEvidenceBundleV3:
        values = _closed(payload, _TOP_LEVEL_FIELDS, "compilation evidence bundle")
        if values.pop("schema") != _SCHEMA:
            raise ValueError("invalid compilation evidence schema")
        target = _closed(values["target"], _TARGET_FIELDS, "compilation target")
        return cls(
            producer=values["producer"],
            version=values["version"],
            source=values["source"],
            target=target,
            physical_plan=PhysicalPlanEvidenceV3.from_dict(
                values["physical_plan"], target_snapshot_id=target["snapshot_id"]
            ),
            output=values["output"],
            bundle_identity=values["bundle_identity"],
        )


__all__ = (
    "COMPILATION_EVIDENCE_VERSION_V3",
    "CompilationEvidenceBundleV3",
    "MappingTransitionEvidenceV3",
    "PhysicalPlanEvidenceV3",
)
