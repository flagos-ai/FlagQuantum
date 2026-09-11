"""Immutable physical mapping and dependency-schedule plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..core.ir import CircuitIR
from ..errors import CompilationError
from .directed_topology import DirectedCouplingMap
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
    physical_to_logical_before: tuple[int | None, ...] = ()
    physical_to_logical_after: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class PhysicalInstructionRecord:
    """One final native instruction linked to source, mapping, and schedule."""

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


def _topology_identity(coupling: CouplingMap | DirectedCouplingMap) -> str:
    if isinstance(coupling, DirectedCouplingMap):
        return coupling.topology_identity
    payload = {
        "n_wires": coupling.n_wires,
        "edges": coupling.edges,
        "direction_semantics": "undirected",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _coupling_allows_instruction(
    coupling: CouplingMap | DirectedCouplingMap,
    opcode: str,
    wires: tuple[int, ...],
) -> bool:
    if len(wires) != 2:
        return True
    if isinstance(coupling, DirectedCouplingMap) and opcode == "swap":
        return coupling.has_weak_edge(*wires)
    return coupling.has_edge(*wires)


def _allocation_identity(
    *,
    logical_wire_count: int,
    physical_slot_count: int,
    initial_logical_to_physical: tuple[int, ...],
    initial_physical_to_logical: tuple[int | None, ...],
    logical_result_physical_slots: tuple[int, ...],
) -> str:
    payload = {
        "logical_wire_count": logical_wire_count,
        "physical_slot_count": physical_slot_count,
        "initial_logical_to_physical": initial_logical_to_physical,
        "initial_physical_to_logical": initial_physical_to_logical,
        "logical_result_physical_slots": logical_result_physical_slots,
        "workspace_initial_state": "standard_zero",
        "workspace_cleanup": "inverse_routing_swaps",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _plan_identity_payload(plan: PhysicalCircuitPlan) -> dict[str, object]:
    allocated = plan.physical_slot_count > plan.logical_wire_count
    instructions: list[dict[str, object]] = [
        {
            "instruction_index": item.instruction_index,
            "source_instruction_index": item.source_instruction_index,
            "topology_instruction_index": item.topology_instruction_index,
            "native_replacement_ordinal": item.native_replacement_ordinal,
            "native_instruction_index": item.native_instruction_index,
            "direction_replacement_ordinal": (item.direction_replacement_ordinal),
            "direction_rewrite": item.direction_rewrite,
            "origin": item.origin,
            "opcode": item.opcode,
            "logical_wires": item.logical_wires,
            "physical_wires": item.physical_wires,
            "layer": item.layer,
            "predecessors": item.predecessors,
            "dependency_kinds": item.dependency_kinds,
        }
        for item in plan.instructions
    ]
    mapping_transitions: list[dict[str, object]] = []
    for item in plan.mapping_transitions:
        transition: dict[str, object] = {
            "routed_instruction_index": item.routed_instruction_index,
            "source_instruction_index": item.source_instruction_index,
            "phase": item.phase,
            "physical_wires": item.physical_wires,
            "layout_before": item.layout_before,
            "layout_after": item.layout_after,
        }
        if allocated:
            transition["physical_to_logical_before"] = item.physical_to_logical_before
            transition["physical_to_logical_after"] = item.physical_to_logical_after
        mapping_transitions.append(transition)

    payload: dict[str, object] = {
        "schema": "flagquantum.physical_circuit_plan",
        "version": plan.version,
        "source_circuit_hash": plan.source_circuit_hash,
        "physical_circuit_hash": plan.program.content_hash,
        "target_snapshot_id": plan.target_snapshot_id,
        "topology_identity": plan.topology_identity,
        "coupling_n_wires": plan.coupling_n_wires,
        "coupling_edges": plan.coupling_edges,
        "coupling_direction_semantics": plan.coupling_direction_semantics,
        "initial_logical_to_physical": plan.initial_logical_to_physical,
        "pre_restore_logical_to_physical": plan.pre_restore_logical_to_physical,
        "final_logical_to_physical": plan.final_logical_to_physical,
        "mapping_transitions": mapping_transitions,
        "instructions": instructions,
        "topology_legalization_identity": plan.topology_legalization_identity,
        "native_gate_legalization_identity": (plan.native_gate_legalization_identity),
        "direction_legalization_identity": plan.direction_legalization_identity,
        "reversed_cx_count": plan.reversed_cx_count,
        "schedule_identity": plan.schedule_identity,
        "schedule_depth": plan.schedule_depth,
        "maximum_parallel_width": plan.maximum_parallel_width,
        "critical_path": plan.critical_path,
    }
    if allocated:
        payload.update(
            {
                "logical_wire_count": plan.logical_wire_count,
                "physical_slot_count": plan.physical_slot_count,
                "initial_physical_to_logical": plan.initial_physical_to_logical,
                "pre_restore_physical_to_logical": (
                    plan.pre_restore_physical_to_logical
                ),
                "final_physical_to_logical": plan.final_physical_to_logical,
                "logical_result_physical_slots": (plan.logical_result_physical_slots),
                "allocation_identity": plan.allocation_identity,
            }
        )
    if plan.coupling_direction_semantics != "directed_cx":
        payload.pop("coupling_direction_semantics")
        payload.pop("direction_legalization_identity")
        payload.pop("reversed_cx_count")
        for instruction in instructions:
            instruction.pop("native_instruction_index")
            instruction.pop("direction_replacement_ordinal")
            instruction.pop("direction_rewrite")
    return payload


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
    coupling_direction_semantics: str
    initial_logical_to_physical: tuple[int, ...]
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    mapping_transitions: tuple[MappingTransition, ...]
    instructions: tuple[PhysicalInstructionRecord, ...]
    topology_legalization_identity: str | None
    native_gate_legalization_identity: str
    direction_legalization_identity: str | None
    reversed_cx_count: int
    schedule_identity: str
    schedule_depth: int
    maximum_parallel_width: int
    critical_path: tuple[int, ...]
    logical_wire_count: int = 0
    physical_slot_count: int = 0
    initial_physical_to_logical: tuple[int | None, ...] = ()
    pre_restore_physical_to_logical: tuple[int | None, ...] = ()
    final_physical_to_logical: tuple[int | None, ...] = ()
    logical_result_physical_slots: tuple[int, ...] = ()
    allocation_identity: str | None = None
    plan_identity: str = ""

    @property
    def version(self) -> str:
        """Return the closed private physical-plan schema version."""

        if self.physical_slot_count > self.logical_wire_count:
            return "3.0"
        if self.coupling_direction_semantics == "directed_cx":
            return "2.0"
        return "1.0"

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
        if self.logical_wire_count != source.n_wires:
            raise PhysicalPlanError("physical plan logical-wire count is inconsistent")
        if self.physical_slot_count != self.program.n_wires:
            raise PhysicalPlanError("physical plan physical-slot count is inconsistent")
        if self.physical_slot_count < self.logical_wire_count:
            raise PhysicalPlanError("physical plan has insufficient physical slots")
        allocated = self.physical_slot_count > self.logical_wire_count
        if (
            len(self.initial_logical_to_physical) != source.n_wires
            or len(set(self.initial_logical_to_physical)) != source.n_wires
            or any(
                physical < 0 or physical >= self.physical_slot_count
                for physical in self.initial_logical_to_physical
            )
        ):
            raise PhysicalPlanError("physical plan initial layout is invalid")
        if not allocated and set(self.initial_logical_to_physical) != set(
            identity_layout
        ):
            raise PhysicalPlanError("physical plan initial layout is invalid")
        if (
            self.coupling_direction_semantics != "directed_cx"
            and self.initial_logical_to_physical != identity_layout
        ):
            raise PhysicalPlanError(
                "undirected physical plan initial layout must be identity"
            )
        self._validate_stage_identities()
        recorded_coupling: CouplingMap | DirectedCouplingMap | None = None
        if topology is None:
            if (
                self.topology_identity is not None
                or self.coupling_n_wires is not None
                or self.coupling_edges
                or self.coupling_direction_semantics != "none"
                or self.mapping_transitions
                or self.pre_restore_logical_to_physical != identity_layout
                or self.final_logical_to_physical != identity_layout
                or allocated
            ):
                raise PhysicalPlanError(
                    "physical plan without topology has mapping evidence"
                )
        else:
            if self.coupling_n_wires is None:
                raise PhysicalPlanError("physical plan lacks coupling-map size")
            if allocated and self.coupling_n_wires != self.physical_slot_count:
                raise PhysicalPlanError(
                    "physical plan coupling size does not match physical slots"
                )
            try:
                if self.coupling_direction_semantics == "directed_cx":
                    recorded_coupling = DirectedCouplingMap(
                        self.coupling_n_wires,
                        self.coupling_edges,
                    )
                elif self.coupling_direction_semantics == "undirected":
                    recorded_coupling = CouplingMap(
                        self.coupling_n_wires,
                        self.coupling_edges,
                    )
                else:
                    raise ValueError("unknown coupling direction semantics")
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
            if allocated:
                if not isinstance(recorded_coupling, DirectedCouplingMap):
                    raise PhysicalPlanError(
                        "physical allocation requires directed topology"
                    )
                self._validate_allocation_evidence()
            self._validate_mapping_transitions(
                allocated=allocated,
                identity_layout=identity_layout,
            )
        if not allocated and (
            self.initial_physical_to_logical
            or self.pre_restore_physical_to_logical
            or self.final_physical_to_logical
            or self.logical_result_physical_slots
            or self.allocation_identity is not None
        ):
            raise PhysicalPlanError("version-1/2 physical plan has allocation evidence")
        self._validate_instruction_records(source, recorded_coupling)
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

    def _validate_stage_identities(self) -> None:
        direction = self.legalization.direction_legalization
        if self.direction_legalization_identity != (
            None if direction is None else direction.legalization_identity
        ):
            raise PhysicalPlanError(
                "physical plan direction legalization identity is inconsistent"
            )
        if self.reversed_cx_count != (
            0 if direction is None else direction.reversed_cx_count
        ):
            raise PhysicalPlanError("physical plan reversed-CX count is inconsistent")
        schedule = self.legalization.schedule
        if self.schedule_depth != schedule.depth:
            raise PhysicalPlanError("physical plan schedule depth is inconsistent")
        if self.maximum_parallel_width != schedule.maximum_parallel_width:
            raise PhysicalPlanError("physical plan schedule width is inconsistent")
        if self.schedule_identity != schedule.schedule_identity:
            raise PhysicalPlanError("physical plan schedule identity is inconsistent")
        if self.native_gate_legalization_identity != (
            self.legalization.native_gate_legalization.legalization_identity
        ):
            raise PhysicalPlanError(
                "physical plan native-gate identity is inconsistent"
            )
        topology = self.legalization.topology_legalization
        expected_topology_identity = (
            None if topology is None else topology.legalization_identity
        )
        if self.topology_legalization_identity != expected_topology_identity:
            raise PhysicalPlanError(
                "physical plan topology legalization identity is inconsistent"
            )

    def _validate_allocation_evidence(self) -> None:
        topology = self.legalization.topology_legalization
        if topology is None:
            raise PhysicalPlanError("physical allocation requires topology evidence")
        expected_occupancy: list[int | None] = [None] * self.physical_slot_count
        for logical, physical in enumerate(self.initial_logical_to_physical):
            expected_occupancy[physical] = logical
        if (
            self.initial_physical_to_logical != tuple(expected_occupancy)
            or self.final_physical_to_logical != self.initial_physical_to_logical
            or len(self.pre_restore_physical_to_logical) != self.physical_slot_count
            or self.logical_result_physical_slots != self.final_logical_to_physical
            or self.logical_result_physical_slots
            != topology.logical_result_physical_slots
            or self.logical_wire_count != topology.logical_wire_count
            or self.physical_slot_count != topology.physical_slot_count
            or self.allocation_identity != topology.allocation_identity
            or self.initial_physical_to_logical != topology.initial_physical_to_logical
            or self.pre_restore_physical_to_logical
            != topology.pre_restore_physical_to_logical
            or self.final_physical_to_logical != topology.final_physical_to_logical
            or self.allocation_identity
            != _allocation_identity(
                logical_wire_count=self.logical_wire_count,
                physical_slot_count=self.physical_slot_count,
                initial_logical_to_physical=self.initial_logical_to_physical,
                initial_physical_to_logical=self.initial_physical_to_logical,
                logical_result_physical_slots=self.logical_result_physical_slots,
            )
        ):
            raise PhysicalPlanError("physical plan allocation evidence is inconsistent")
        if any(
            measurement.wires != self.logical_result_physical_slots
            or str(measurement.metadata.get("fq_output_kind", measurement.kind))
            != "samples"
            or measurement.metadata.get("fq_result_order") != "logical_wire_order"
            for measurement in self.program.measurements
        ):
            raise PhysicalPlanError(
                "physical plan logical result projection is inconsistent"
            )

    def _validate_mapping_transitions(
        self,
        *,
        allocated: bool,
        identity_layout: tuple[int, ...],
    ) -> None:
        layout = list(self.initial_logical_to_physical)
        occupancy = list(self.initial_physical_to_logical)
        for transition in self.mapping_transitions:
            if transition.layout_before != tuple(layout):
                raise PhysicalPlanError(
                    "physical plan mapping transitions are not continuous"
                )
            if allocated:
                if transition.physical_to_logical_before != tuple(occupancy):
                    raise PhysicalPlanError(
                        "physical plan occupancy transitions are not continuous"
                    )
                _apply_allocated_physical_swap(
                    layout,
                    occupancy,
                    transition.physical_wires,
                )
                if transition.physical_to_logical_after != tuple(occupancy):
                    raise PhysicalPlanError(
                        "physical plan occupancy transition result is inconsistent"
                    )
            else:
                if (
                    transition.physical_to_logical_before
                    or transition.physical_to_logical_after
                ):
                    raise PhysicalPlanError(
                        "version-1/2 mapping transition has occupancy evidence"
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
        if allocated and tuple(occupancy) != self.final_physical_to_logical:
            raise PhysicalPlanError(
                "physical plan occupancy transitions do not reach final occupancy"
            )
        if not allocated and self.final_logical_to_physical != identity_layout:
            raise PhysicalPlanError("physical plan final layout must be identity")

    def _validate_instruction_records(
        self,
        source: CircuitIR,
        coupling: CouplingMap | DirectedCouplingMap | None,
    ) -> None:
        if len(self.instructions) != len(self.program.instructions):
            raise PhysicalPlanError("physical plan instruction count is inconsistent")
        for index, item in enumerate(self.instructions):
            self._validate_instruction_record(index, item, source, coupling)

    def _validate_instruction_record(
        self,
        index: int,
        item: PhysicalInstructionRecord,
        source: CircuitIR,
        coupling: CouplingMap | DirectedCouplingMap | None,
    ) -> None:
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
        if coupling is not None and not _coupling_allows_instruction(
            coupling, instruction.name, instruction.wires
        ):
            raise PhysicalPlanError(
                f"physical instruction record {index} violates topology"
            )
        if item.native_instruction_index < 0:
            raise PhysicalPlanError(
                f"physical instruction record {index} has invalid native index"
            )
        if item.direction_replacement_ordinal < 0:
            raise PhysicalPlanError(
                f"physical instruction record {index} has invalid direction ordinal"
            )
        if item.direction_rewrite not in {"none", "reverse_cx_h_conjugation"}:
            raise PhysicalPlanError(
                f"physical instruction record {index} has invalid direction rewrite"
            )
        expected_native_index = instruction.metadata.get(
            "native_instruction_index", index
        )
        expected_direction_ordinal = instruction.metadata.get(
            "direction_replacement_ordinal", 0
        )
        expected_direction_rewrite = instruction.metadata.get(
            "direction_rewrite", "none"
        )
        if (
            item.native_instruction_index != expected_native_index
            or item.direction_replacement_ordinal != expected_direction_ordinal
            or item.direction_rewrite != expected_direction_rewrite
        ):
            raise PhysicalPlanError(
                f"physical instruction record {index} has inconsistent direction lineage"
            )


def _apply_physical_swap(layout: list[int], wires: tuple[int, int]) -> None:
    left, right = wires
    inverse = {physical: logical for logical, physical in enumerate(layout)}
    try:
        left_logical = inverse[left]
        right_logical = inverse[right]
    except KeyError as error:
        raise PhysicalPlanError("routing SWAP references an unmapped wire") from error
    layout[left_logical], layout[right_logical] = right, left


def _apply_allocated_physical_swap(
    layout: list[int],
    occupancy: list[int | None],
    wires: tuple[int, int],
) -> None:
    left, right = wires
    if left == right or min(left, right) < 0 or max(left, right) >= len(occupancy):
        raise PhysicalPlanError("routing SWAP references an invalid physical slot")
    left_logical = occupancy[left]
    right_logical = occupancy[right]
    occupancy[left], occupancy[right] = right_logical, left_logical
    if left_logical is not None:
        layout[left_logical] = right
    if right_logical is not None:
        layout[right_logical] = left


def _mapping_evidence(
    source: CircuitIR,
    routed: CircuitIR,
    *,
    initial_layout: tuple[int, ...],
    final_layout: tuple[int, ...],
    initial_occupancy: tuple[int | None, ...] = (),
    final_occupancy: tuple[int | None, ...] = (),
) -> tuple[
    tuple[MappingTransition, ...],
    tuple[int, ...],
    tuple[int | None, ...],
]:
    layout = list(initial_layout)
    allocated = bool(initial_occupancy)
    occupancy = list(initial_occupancy)
    transitions = []
    pre_restore_layout = tuple(layout)
    pre_restore_occupancy = tuple(occupancy)
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
                pre_restore_occupancy = tuple(occupancy)
                saw_restore = True
            before = tuple(layout)
            occupancy_before: tuple[int | None, ...] = ()
            occupancy_after: tuple[int | None, ...] = ()
            if allocated:
                occupancy_before = tuple(occupancy)
                if (
                    instruction.metadata.get("physical_to_logical_before")
                    != occupancy_before
                    or instruction.metadata.get("logical_to_physical_before") != before
                ):
                    raise PhysicalPlanError(
                        f"routed instruction {index} has inconsistent allocation input"
                    )
                _apply_allocated_physical_swap(
                    layout,
                    occupancy,
                    instruction.wires,
                )
                occupancy_after = tuple(occupancy)
                if instruction.metadata.get(
                    "physical_to_logical_after"
                ) != occupancy_after or instruction.metadata.get(
                    "logical_to_physical_after"
                ) != tuple(
                    layout
                ):
                    raise PhysicalPlanError(
                        f"routed instruction {index} has inconsistent allocation output"
                    )
            else:
                _apply_physical_swap(layout, instruction.wires)
            transitions.append(
                MappingTransition(
                    routed_instruction_index=index,
                    source_instruction_index=raw_source_index,
                    phase=str(phase),
                    physical_wires=instruction.wires,
                    layout_before=before,
                    layout_after=tuple(layout),
                    physical_to_logical_before=occupancy_before,
                    physical_to_logical_after=occupancy_after,
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
    if allocated and tuple(occupancy) != final_occupancy:
        raise PhysicalPlanError(
            "routing transition replay does not reach final occupancy"
        )
    if not saw_restore:
        pre_restore_layout = tuple(layout)
        pre_restore_occupancy = tuple(occupancy)
    return tuple(transitions), pre_restore_layout, pre_restore_occupancy


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
    coupling_map: CouplingMap | DirectedCouplingMap | None = None,
) -> PhysicalCircuitPlan:
    """Compose verified mapping, native-gate, and schedule evidence."""

    if not isinstance(legalization, TargetLegalizationResult):
        raise TypeError("physical plan requires a TargetLegalizationResult")
    topology = legalization.topology_legalization
    native = legalization.native_gate_legalization
    direction = legalization.direction_legalization
    schedule = legalization.schedule
    if schedule.program is not legalization.program:
        raise PhysicalPlanError(
            "physical plan stages do not share the legalized CircuitIR"
        )
    if direction is None and native.program is not legalization.program:
        raise PhysicalPlanError("native legalization does not own final CircuitIR")
    if direction is not None and (
        direction.source_program is not native.program
        or direction.program is not legalization.program
    ):
        raise PhysicalPlanError("direction legalization stage chain is inconsistent")

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
        coupling_direction_semantics = "none"
        initial_layout = final_layout = tuple(range(source.n_wires))
        transitions: tuple[MappingTransition, ...] = ()
        pre_restore_layout = final_layout
        initial_occupancy: tuple[int | None, ...] = ()
        pre_restore_occupancy: tuple[int | None, ...] = ()
        final_occupancy: tuple[int | None, ...] = ()
        result_slots: tuple[int, ...] = ()
        allocation_identity = None
        source_circuit_hash = native.source_content_hash
    else:
        if not isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
            raise PhysicalPlanError(
                "topology-legalized plans require the exact coupling map"
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
        coupling_direction_semantics = (
            "directed_cx"
            if isinstance(coupling_map, DirectedCouplingMap)
            else "undirected"
        )
        initial_layout = topology.initial_logical_to_physical
        final_layout = topology.final_logical_to_physical
        initial_occupancy = topology.initial_physical_to_logical
        final_occupancy = topology.final_physical_to_logical
        result_slots = topology.logical_result_physical_slots
        allocation_identity = topology.allocation_identity
        source_circuit_hash = topology.source_content_hash
        if native.source_program is not routed:
            raise PhysicalPlanError(
                "native legalization does not retain the topology result"
            )
        transitions, replay_pre_restore, replay_pre_restore_occupancy = (
            _mapping_evidence(
                source,
                routed,
                initial_layout=initial_layout,
                final_layout=final_layout,
                initial_occupancy=initial_occupancy,
                final_occupancy=final_occupancy,
            )
        )
        routing_metadata = routed.metadata.get("routing")
        if not isinstance(routing_metadata, dict):
            raise PhysicalPlanError("routed CircuitIR lacks routing metadata")
        pre_restore_layout = tuple(routing_metadata["pre_restore_logical_to_physical"])
        if replay_pre_restore != pre_restore_layout:
            raise PhysicalPlanError("routing pre-restore layout is inconsistent")
        pre_restore_occupancy = tuple(
            routing_metadata.get("pre_restore_physical_to_logical", ())
        )
        if replay_pre_restore_occupancy != pre_restore_occupancy:
            raise PhysicalPlanError(
                "routing pre-restore physical occupancy is inconsistent"
            )

    decompositions = {item.instruction_index: item for item in native.decompositions}
    native_lineage: dict[int, tuple[int, int]] = {}
    native_index = 0
    for topology_index in range(len(routed.instructions)):
        decomposition = decompositions.get(topology_index)
        replacement_count = (
            1 if decomposition is None else len(decomposition.replacement_opcodes)
        )
        for ordinal in range(replacement_count):
            native_lineage[native_index] = (topology_index, ordinal)
            native_index += 1
    if native_index != len(native.program.instructions):
        raise PhysicalPlanError("native decomposition records do not cover the program")
    schedule_by_index = {item.instruction_index: item for item in schedule.instructions}
    instruction_records = []
    for physical_index, instruction in enumerate(legalization.program.instructions):
        raw_native_index = instruction.metadata.get(
            "native_instruction_index", physical_index
        )
        if type(raw_native_index) is not int or raw_native_index not in native_lineage:
            raise PhysicalPlanError("physical instruction native index is invalid")
        topology_index, native_ordinal = native_lineage[raw_native_index]
        topology_instruction = routed.instructions[topology_index]
        decomposition = decompositions.get(topology_index)
        raw_source_index = topology_instruction.metadata.get(
            "source_instruction_index", topology_index
        )
        if type(raw_source_index) is not int:
            raise PhysicalPlanError("physical instruction source index is invalid")
        if not 0 <= raw_source_index < len(source.instructions):
            raise PhysicalPlanError("physical instruction source index is out of range")
        logical_wires = source.instructions[raw_source_index].wires
        scheduled = schedule_by_index[physical_index]
        raw_direction_ordinal = instruction.metadata.get(
            "direction_replacement_ordinal", 0
        )
        raw_direction_rewrite = instruction.metadata.get("direction_rewrite", "none")
        if type(raw_direction_ordinal) is not int:
            raise PhysicalPlanError("physical instruction direction ordinal is invalid")
        if raw_direction_rewrite == "reverse_cx_h_conjugation":
            origin = (
                "routing_swap_direction_rewrite"
                if topology_instruction.metadata.get("routing_phase") is not None
                else "direction_rewrite"
            )
        elif topology_instruction.metadata.get("routing_phase") is not None:
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
                native_replacement_ordinal=native_ordinal,
                native_instruction_index=raw_native_index,
                direction_replacement_ordinal=raw_direction_ordinal,
                direction_rewrite=str(raw_direction_rewrite),
                origin=origin,
                opcode=instruction.name,
                logical_wires=logical_wires,
                physical_wires=instruction.wires,
                layer=scheduled.layer,
                predecessors=scheduled.predecessors,
                dependency_kinds=scheduled.dependency_kinds,
            )
        )

    if topology is not None:
        assert coupling_map is not None
        for index, instruction in enumerate(legalization.program.instructions):
            if not _coupling_allows_instruction(
                coupling_map, instruction.name, instruction.wires
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
        coupling_direction_semantics=coupling_direction_semantics,
        initial_logical_to_physical=initial_layout,
        pre_restore_logical_to_physical=pre_restore_layout,
        final_logical_to_physical=final_layout,
        mapping_transitions=transitions,
        instructions=tuple(instruction_records),
        topology_legalization_identity=topology_legalization_identity,
        native_gate_legalization_identity=native.legalization_identity,
        direction_legalization_identity=(
            None if direction is None else direction.legalization_identity
        ),
        reversed_cx_count=(0 if direction is None else direction.reversed_cx_count),
        schedule_identity=schedule.schedule_identity,
        schedule_depth=schedule.depth,
        maximum_parallel_width=schedule.maximum_parallel_width,
        critical_path=_critical_path(legalization),
        logical_wire_count=source.n_wires,
        physical_slot_count=legalization.program.n_wires,
        initial_physical_to_logical=initial_occupancy,
        pre_restore_physical_to_logical=pre_restore_occupancy,
        final_physical_to_logical=final_occupancy,
        logical_result_physical_slots=result_slots,
        allocation_identity=allocation_identity,
    )


__all__ = (
    "MappingTransition",
    "PhysicalCircuitPlan",
    "PhysicalInstructionRecord",
    "PhysicalPlanError",
    "build_physical_circuit_plan",
)
