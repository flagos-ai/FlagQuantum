"""Immutable physical mapping and dependency-schedule plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..core.ir import CircuitIR
from ..errors import CompilationError
from .routing import CouplingMap
from .target_legalization import TargetLegalizationResult


class PhysicalPlanError(CompilationError):
    """Target legalization evidence cannot form a physical circuit plan."""


@dataclass(frozen=True)
class MappingTransition:
    """One routing SWAP and its logical-to-physical layout transition."""

    routed_instruction_index: int
    source_instruction_index: int
    phase: str
    physical_wires: tuple[int, int]
    layout_before: tuple[int, ...]
    layout_after: tuple[int, ...]


@dataclass(frozen=True)
class PhysicalInstructionRecord:
    """One final native instruction linked to source, mapping, and schedule."""

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


def _topology_identity(coupling: CouplingMap) -> str:
    payload = {
        "n_wires": coupling.n_wires,
        "edges": coupling.edges,
        "direction_semantics": "undirected",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _plan_identity_payload(plan: PhysicalCircuitPlan) -> dict[str, object]:
    return {
        "schema": "flagquantum.physical_circuit_plan",
        "version": "1.0",
        "source_circuit_hash": plan.source_circuit_hash,
        "physical_circuit_hash": plan.program.content_hash,
        "target_snapshot_id": plan.target_snapshot_id,
        "topology_identity": plan.topology_identity,
        "coupling_n_wires": plan.coupling_n_wires,
        "coupling_edges": plan.coupling_edges,
        "initial_logical_to_physical": plan.initial_logical_to_physical,
        "pre_restore_logical_to_physical": plan.pre_restore_logical_to_physical,
        "final_logical_to_physical": plan.final_logical_to_physical,
        "mapping_transitions": [
            {
                "routed_instruction_index": item.routed_instruction_index,
                "source_instruction_index": item.source_instruction_index,
                "phase": item.phase,
                "physical_wires": item.physical_wires,
                "layout_before": item.layout_before,
                "layout_after": item.layout_after,
            }
            for item in plan.mapping_transitions
        ],
        "instructions": [
            {
                "instruction_index": item.instruction_index,
                "source_instruction_index": item.source_instruction_index,
                "topology_instruction_index": item.topology_instruction_index,
                "native_replacement_ordinal": item.native_replacement_ordinal,
                "origin": item.origin,
                "opcode": item.opcode,
                "logical_wires": item.logical_wires,
                "physical_wires": item.physical_wires,
                "layer": item.layer,
                "predecessors": item.predecessors,
                "dependency_kinds": item.dependency_kinds,
            }
            for item in plan.instructions
        ],
        "topology_legalization_identity": plan.topology_legalization_identity,
        "native_gate_legalization_identity": (plan.native_gate_legalization_identity),
        "schedule_identity": plan.schedule_identity,
        "schedule_depth": plan.schedule_depth,
        "maximum_parallel_width": plan.maximum_parallel_width,
        "critical_path": plan.critical_path,
    }


@dataclass(frozen=True)
class PhysicalCircuitPlan:
    """A physical mapping/schedule view over the authoritative Core CircuitIR."""

    legalization: TargetLegalizationResult = field(repr=False)
    program: CircuitIR
    source_circuit_hash: str
    target_snapshot_id: str
    topology_identity: str | None
    coupling_n_wires: int | None
    coupling_edges: tuple[tuple[int, int], ...]
    initial_logical_to_physical: tuple[int, ...]
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    mapping_transitions: tuple[MappingTransition, ...]
    instructions: tuple[PhysicalInstructionRecord, ...]
    topology_legalization_identity: str | None
    native_gate_legalization_identity: str
    schedule_identity: str
    schedule_depth: int
    maximum_parallel_width: int
    critical_path: tuple[int, ...]
    plan_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.legalization, TargetLegalizationResult):
            raise TypeError("physical plan requires a TargetLegalizationResult")
        if self.program is not self.legalization.program:
            raise PhysicalPlanError(
                "physical plan program must be the legalized CircuitIR instance"
            )
        if self.target_snapshot_id != self.legalization.target_snapshot_id:
            raise PhysicalPlanError("physical plan target snapshot is inconsistent")
        topology = self.legalization.topology_legalization
        expected_source_hash = (
            self.legalization.native_gate_legalization.source_content_hash
            if topology is None
            else topology.source_content_hash
        )
        if self.source_circuit_hash != expected_source_hash:
            raise PhysicalPlanError("physical plan source circuit is inconsistent")
        source = (
            self.legalization.native_gate_legalization.source_program
            if topology is None
            else topology.source_program
        )
        identity_layout = tuple(range(source.n_wires))
        if self.initial_logical_to_physical != identity_layout:
            raise PhysicalPlanError("physical plan initial layout must be identity")
        if topology is None:
            if (
                self.topology_identity is not None
                or self.coupling_n_wires is not None
                or self.coupling_edges
                or self.mapping_transitions
                or self.pre_restore_logical_to_physical != identity_layout
                or self.final_logical_to_physical != identity_layout
            ):
                raise PhysicalPlanError(
                    "physical plan without topology has mapping evidence"
                )
        else:
            if self.coupling_n_wires is None:
                raise PhysicalPlanError("physical plan lacks coupling-map size")
            try:
                recorded_coupling = CouplingMap(
                    self.coupling_n_wires,
                    self.coupling_edges,
                )
            except ValueError as error:
                raise PhysicalPlanError(
                    "physical plan coupling map is invalid"
                ) from error
            if _topology_identity(recorded_coupling) != self.topology_identity:
                raise PhysicalPlanError(
                    "physical plan coupling map does not match topology identity"
                )
            if self.final_logical_to_physical != topology.final_logical_to_physical:
                raise PhysicalPlanError("physical plan final layout is inconsistent")
            layout = list(self.initial_logical_to_physical)
            for transition in self.mapping_transitions:
                if transition.layout_before != tuple(layout):
                    raise PhysicalPlanError(
                        "physical plan mapping transitions are not continuous"
                    )
                _apply_physical_swap(layout, transition.physical_wires)
                if transition.layout_after != tuple(layout):
                    raise PhysicalPlanError(
                        "physical plan mapping transition result is inconsistent"
                    )
            if tuple(layout) != self.final_logical_to_physical:
                raise PhysicalPlanError(
                    "physical plan mapping transitions do not reach final layout"
                )
        if self.schedule_depth != self.legalization.schedule.depth:
            raise PhysicalPlanError("physical plan schedule depth is inconsistent")
        if (
            self.maximum_parallel_width
            != self.legalization.schedule.maximum_parallel_width
        ):
            raise PhysicalPlanError("physical plan schedule width is inconsistent")
        if self.schedule_identity != self.legalization.schedule.schedule_identity:
            raise PhysicalPlanError("physical plan schedule identity is inconsistent")
        if self.native_gate_legalization_identity != (
            self.legalization.native_gate_legalization.legalization_identity
        ):
            raise PhysicalPlanError(
                "physical plan native-gate identity is inconsistent"
            )
        expected_topology_identity = (
            None if topology is None else topology.legalization_identity
        )
        if self.topology_legalization_identity != expected_topology_identity:
            raise PhysicalPlanError(
                "physical plan topology legalization identity is inconsistent"
            )
        if len(self.instructions) != len(self.program.instructions):
            raise PhysicalPlanError("physical plan instruction count is inconsistent")
        for index, item in enumerate(self.instructions):
            scheduled = self.legalization.schedule.instructions[index]
            instruction = self.program.instructions[index]
            if (
                item.instruction_index != index
                or item.opcode != instruction.name
                or item.physical_wires != instruction.wires
                or item.layer != scheduled.layer
                or item.predecessors != scheduled.predecessors
                or item.dependency_kinds != scheduled.dependency_kinds
            ):
                raise PhysicalPlanError(
                    f"physical instruction record {index} is inconsistent"
                )
            if not 0 <= item.source_instruction_index < len(source.instructions):
                raise PhysicalPlanError(
                    f"physical instruction record {index} has invalid source index"
                )
            if (
                item.logical_wires
                != source.instructions[item.source_instruction_index].wires
            ):
                raise PhysicalPlanError(
                    f"physical instruction record {index} has invalid logical wires"
                )
            if topology is not None and len(instruction.wires) == 2:
                assert self.coupling_n_wires is not None
                if not recorded_coupling.has_edge(*instruction.wires):
                    raise PhysicalPlanError(
                        f"physical instruction record {index} violates topology"
                    )
        if self.critical_path != _critical_path(self.legalization):
            raise PhysicalPlanError("physical plan critical path is inconsistent")
        expected_identity = hashlib.sha256(
            json.dumps(
                _plan_identity_payload(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.plan_identity and self.plan_identity != expected_identity:
            raise PhysicalPlanError("plan_identity does not match physical plan")
        object.__setattr__(self, "plan_identity", expected_identity)


def _apply_physical_swap(layout: list[int], wires: tuple[int, int]) -> None:
    left, right = wires
    inverse = {physical: logical for logical, physical in enumerate(layout)}
    try:
        left_logical = inverse[left]
        right_logical = inverse[right]
    except KeyError as error:
        raise PhysicalPlanError("routing SWAP references an unmapped wire") from error
    layout[left_logical], layout[right_logical] = right, left


def _mapping_evidence(
    source: CircuitIR,
    routed: CircuitIR,
    *,
    final_layout: tuple[int, ...],
) -> tuple[
    tuple[MappingTransition, ...],
    tuple[int, ...],
]:
    layout = list(range(source.n_wires))
    transitions = []
    pre_restore_layout = tuple(layout)
    saw_restore = False
    for index, instruction in enumerate(routed.instructions):
        raw_source_index = instruction.metadata.get("source_instruction_index")
        if type(raw_source_index) is not int or not 0 <= raw_source_index < len(
            source.instructions
        ):
            raise PhysicalPlanError(
                f"routed instruction {index} has no valid source instruction"
            )
        phase = instruction.metadata.get("routing_phase")
        if phase is not None:
            if instruction.name != "swap" or len(instruction.wires) != 2:
                raise PhysicalPlanError(
                    f"routed instruction {index} has invalid SWAP evidence"
                )
            if phase == "final_restore" and not saw_restore:
                pre_restore_layout = tuple(layout)
                saw_restore = True
            before = tuple(layout)
            _apply_physical_swap(layout, instruction.wires)
            transitions.append(
                MappingTransition(
                    routed_instruction_index=index,
                    source_instruction_index=raw_source_index,
                    phase=str(phase),
                    physical_wires=instruction.wires,
                    layout_before=before,
                    layout_after=tuple(layout),
                )
            )
            continue
        logical_wires = tuple(
            instruction.metadata.get("logical_wires", instruction.wires)
        )
        mapped_wires = tuple(layout[wire] for wire in logical_wires)
        if mapped_wires != instruction.wires:
            raise PhysicalPlanError(
                f"routed instruction {index} does not match the active layout"
            )
    if tuple(layout) != final_layout:
        raise PhysicalPlanError("routing transition replay does not reach final layout")
    if not saw_restore:
        pre_restore_layout = tuple(layout)
    return tuple(transitions), pre_restore_layout


def _critical_path(legalization: TargetLegalizationResult) -> tuple[int, ...]:
    records = legalization.schedule.instructions
    if not records:
        return ()
    by_index = {item.instruction_index: item for item in records}
    current = min(
        (item for item in records if item.layer == legalization.schedule.depth - 1),
        key=lambda item: item.instruction_index,
    )
    reversed_path = [current.instruction_index]
    while current.predecessors:
        maximum_layer = max(by_index[index].layer for index in current.predecessors)
        predecessor = min(
            index
            for index in current.predecessors
            if by_index[index].layer == maximum_layer
        )
        reversed_path.append(predecessor)
        current = by_index[predecessor]
    return tuple(reversed(reversed_path))


def build_physical_circuit_plan(
    legalization: TargetLegalizationResult,
    *,
    coupling_map: CouplingMap | None = None,
) -> PhysicalCircuitPlan:
    """Compose verified mapping, native-gate, and schedule evidence."""

    if not isinstance(legalization, TargetLegalizationResult):
        raise TypeError("physical plan requires a TargetLegalizationResult")
    topology = legalization.topology_legalization
    native = legalization.native_gate_legalization
    schedule = legalization.schedule
    if schedule.program is not legalization.program or native.program is not (
        legalization.program
    ):
        raise PhysicalPlanError(
            "physical plan stages do not share the legalized CircuitIR"
        )

    if topology is None:
        if coupling_map is not None:
            raise PhysicalPlanError(
                "a coupling map cannot be attached after topology legalization"
            )
        source = native.source_program
        routed = source
        topology_identity = None
        coupling_n_wires = None
        topology_legalization_identity = None
        coupling_edges: tuple[tuple[int, int], ...] = ()
        initial_layout = final_layout = tuple(range(source.n_wires))
        transitions: tuple[MappingTransition, ...] = ()
        pre_restore_layout = final_layout
        source_circuit_hash = native.source_content_hash
    else:
        if not isinstance(coupling_map, CouplingMap):
            raise PhysicalPlanError(
                "topology-legalized plans require the exact CouplingMap"
            )
        if _topology_identity(coupling_map) != topology.topology_identity:
            raise PhysicalPlanError(
                "coupling map does not match topology legalization identity"
            )
        source = topology.source_program
        routed = topology.program
        topology_identity = topology.topology_identity
        coupling_n_wires = coupling_map.n_wires
        topology_legalization_identity = topology.legalization_identity
        coupling_edges = coupling_map.edges
        initial_layout = tuple(range(routed.n_wires))
        final_layout = topology.final_logical_to_physical
        source_circuit_hash = topology.source_content_hash
        if native.source_program is not routed:
            raise PhysicalPlanError(
                "native legalization does not retain the topology result"
            )
        transitions, replay_pre_restore = _mapping_evidence(
            source, routed, final_layout=final_layout
        )
        routing_metadata = routed.metadata.get("routing")
        if not isinstance(routing_metadata, dict):
            raise PhysicalPlanError("routed CircuitIR lacks routing metadata")
        pre_restore_layout = tuple(routing_metadata["pre_restore_logical_to_physical"])
        if replay_pre_restore != pre_restore_layout:
            raise PhysicalPlanError("routing pre-restore layout is inconsistent")

    decompositions = {item.instruction_index: item for item in native.decompositions}
    schedule_by_index = {item.instruction_index: item for item in schedule.instructions}
    instruction_records = []
    physical_index = 0
    for topology_index, topology_instruction in enumerate(routed.instructions):
        decomposition = decompositions.get(topology_index)
        replacement_count = (
            1 if decomposition is None else len(decomposition.replacement_opcodes)
        )
        raw_source_index = topology_instruction.metadata.get(
            "source_instruction_index", topology_index
        )
        if type(raw_source_index) is not int:
            raise PhysicalPlanError("physical instruction source index is invalid")
        if not 0 <= raw_source_index < len(source.instructions):
            raise PhysicalPlanError("physical instruction source index is out of range")
        logical_wires = source.instructions[raw_source_index].wires
        for ordinal in range(replacement_count):
            instruction = legalization.program.instructions[physical_index]
            scheduled = schedule_by_index[physical_index]
            if topology_instruction.metadata.get("routing_phase") is not None:
                origin = (
                    "routing_swap"
                    if decomposition is None
                    else "routing_swap_decomposition"
                )
            elif decomposition is not None:
                origin = "native_decomposition"
            elif topology is not None:
                origin = "topology_mapped"
            else:
                origin = "source"
            instruction_records.append(
                PhysicalInstructionRecord(
                    instruction_index=physical_index,
                    source_instruction_index=raw_source_index,
                    topology_instruction_index=topology_index,
                    native_replacement_ordinal=ordinal,
                    origin=origin,
                    opcode=instruction.name,
                    logical_wires=logical_wires,
                    physical_wires=instruction.wires,
                    layer=scheduled.layer,
                    predecessors=scheduled.predecessors,
                    dependency_kinds=scheduled.dependency_kinds,
                )
            )
            physical_index += 1
    if physical_index != len(legalization.program.instructions):
        raise PhysicalPlanError("native decomposition records do not cover the program")

    if topology is not None:
        assert coupling_map is not None
        for index, instruction in enumerate(legalization.program.instructions):
            if len(instruction.wires) == 2 and not coupling_map.has_edge(
                *instruction.wires
            ):
                raise PhysicalPlanError(
                    f"physical instruction {index} violates the coupling topology"
                )

    return PhysicalCircuitPlan(
        legalization=legalization,
        program=legalization.program,
        source_circuit_hash=source_circuit_hash,
        target_snapshot_id=legalization.target_snapshot_id,
        topology_identity=topology_identity,
        coupling_n_wires=coupling_n_wires,
        coupling_edges=coupling_edges,
        initial_logical_to_physical=initial_layout,
        pre_restore_logical_to_physical=pre_restore_layout,
        final_logical_to_physical=final_layout,
        mapping_transitions=transitions,
        instructions=tuple(instruction_records),
        topology_legalization_identity=topology_legalization_identity,
        native_gate_legalization_identity=native.legalization_identity,
        schedule_identity=schedule.schedule_identity,
        schedule_depth=schedule.depth,
        maximum_parallel_width=schedule.maximum_parallel_width,
        critical_path=_critical_path(legalization),
    )


__all__ = (
    "MappingTransition",
    "PhysicalCircuitPlan",
    "PhysicalInstructionRecord",
    "PhysicalPlanError",
    "build_physical_circuit_plan",
)
