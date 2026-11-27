"""Strict many-to-many compilation evidence; no numerical equivalence oracle."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from flagquantum.core.operators.schema import get_operator_schema

from ._compilation_evidence import (
    _DEPENDENCY_KINDS,
    _MAX_BUNDLE_BYTES,
    _MAX_PREDECESSORS,
    _MAX_RECORDS,
    _ORIGINS,
    _OUTPUT_FIELDS,
    _PLAN_FIELDS,
    _SCHEMA,
    _SOURCE_FIELDS,
    _TARGET_FIELDS,
    _TOP_LEVEL_FIELDS,
    CouplingEvidence,
    MappingTransitionEvidence,
    _bounded_string,
    _canonical_bytes,
    _closed,
    _coupling_identity,
    _index,
    _int_tuple,
    _sha256,
    _swap_layout,
)

_NATIVE_FIELDS = {
    "instruction_index",
    "source_instruction_index",
    "topology_instruction_index",
    "native_replacement_ordinal",
    "origin",
    "opcode",
    "source_logical_qubits",
    "physical_qubits",
}
_REWRITE_FIELDS = {
    "input_instruction_indices",
    "output_instruction_indices",
    "physical_qubits",
    "baseline_cz_count",
    "replacement_cz_count",
    "equivalence",
    "phase_real",
    "phase_imag",
    "max_abs_residual",
}
_INSTRUCTION_FIELDS = {
    "instruction_index",
    "native_input_instruction_index",
    "block_rewrite_index",
    "block_replacement_ordinal",
    "opcode",
    "physical_qubits",
    "layer",
    "predecessors",
    "dependency_kinds",
}
_PLAN_FIELDS_V4 = _PLAN_FIELDS | {
    "native_input_circuit_hash",
    "native_input_instructions",
    "block_rewrites",
    "native_block_optimization_identity",
    "mapping_transition_space",
}


def _array(value: object, owner: str, limit: int = _MAX_RECORDS) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{owner} must be an array")
    if len(value) > limit:
        raise ValueError(f"{owner} exceeds record limit {limit}")
    return tuple(value)


def _indices(value: object, owner: str, limit: int = _MAX_RECORDS) -> tuple[int, ...]:
    _array(value, owner, limit)
    result = _int_tuple(value, owner)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{owner} must be sorted and unique")
    return result


def _qubits(value: object, owner: str) -> tuple[int, ...]:
    _array(value, owner)
    result = _int_tuple(value, owner)
    if len(result) > _MAX_RECORDS or len(set(result)) != len(result):
        raise ValueError(f"{owner} must be bounded and unique")
    return result


def _opcode(opcode: str, qubits: tuple[int, ...]) -> None:
    _bounded_string(opcode, "opcode")
    schema = get_operator_schema(opcode)
    if schema is None or schema.opcode != opcode:
        raise ValueError("instruction must name a known canonical opcode")
    if schema.arity >= 0 and schema.arity != len(qubits):
        raise ValueError("opcode arity does not match physical qubits")


@dataclass(frozen=True)
class NativeInputInstructionEvidence:
    """One native baseline instruction with its retained source ancestry."""

    instruction_index: int
    source_instruction_index: int
    topology_instruction_index: int
    native_replacement_ordinal: int
    origin: str
    opcode: str
    source_logical_qubits: tuple[int, ...]
    physical_qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "instruction_index",
            "source_instruction_index",
            "topology_instruction_index",
            "native_replacement_ordinal",
        ):
            _index(getattr(self, name), name)
        if self.origin not in _ORIGINS:
            raise ValueError("native input has unsupported origin")
        logical = _qubits(self.source_logical_qubits, "source_logical_qubits")
        physical = _qubits(self.physical_qubits, "physical_qubits")
        _opcode(self.opcode, physical)
        object.__setattr__(self, "source_logical_qubits", logical)
        object.__setattr__(self, "physical_qubits", physical)

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction_index": self.instruction_index,
            "source_instruction_index": self.source_instruction_index,
            "topology_instruction_index": self.topology_instruction_index,
            "native_replacement_ordinal": self.native_replacement_ordinal,
            "origin": self.origin,
            "opcode": self.opcode,
            "source_logical_qubits": list(self.source_logical_qubits),
            "physical_qubits": list(self.physical_qubits),
        }

    @classmethod
    def from_dict(cls, payload: object) -> NativeInputInstructionEvidence:
        return cls(**_closed(payload, _NATIVE_FIELDS, "native input instruction"))


@dataclass(frozen=True)
class BlockRewriteEvidence:
    """Bounded block membership and a claimed numerical equivalence witness."""

    input_instruction_indices: tuple[int, ...]
    output_instruction_indices: tuple[int, ...]
    physical_qubits: tuple[int, int]
    baseline_cz_count: int
    replacement_cz_count: int
    equivalence: str
    phase_real: float
    phase_imag: float
    max_abs_residual: float

    def __post_init__(self) -> None:
        inputs = _indices(self.input_instruction_indices, "block inputs", 64)
        outputs = _indices(self.output_instruction_indices, "block outputs", 96)
        pair = _qubits(self.physical_qubits, "block physical_qubits")
        if not inputs or len(pair) != 2:
            raise ValueError("block requires nonempty inputs and two distinct qubits")
        if outputs and outputs != tuple(range(outputs[0], outputs[0] + len(outputs))):
            raise ValueError("block output indices must be contiguous")
        if len(outputs) > len(inputs) + 32:
            raise ValueError("block exceeds additional instruction budget")
        before = _index(self.baseline_cz_count, "baseline_cz_count")
        after = _index(self.replacement_cz_count, "replacement_cz_count")
        if before > len(inputs) or after > len(outputs) or after >= before:
            raise ValueError("block must strictly reduce CZ count")
        if self.equivalence != "unitary_up_to_global_phase":
            raise ValueError("block equivalence must be unitary_up_to_global_phase")
        for name in ("phase_real", "phase_imag", "max_abs_residual"):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise ValueError(f"{name} must be a finite real scalar")
            try:
                finite = math.isfinite(value)
            except OverflowError as error:
                raise ValueError(f"{name} exceeds finite scalar range") from error
            if not finite:
                raise ValueError(f"{name} must be a finite real scalar")
        if abs(math.hypot(self.phase_real, self.phase_imag) - 1) > 1e-11:
            raise ValueError("block phase modulus exceeds residual bound")
        if not 0 <= self.max_abs_residual <= 1e-11:
            raise ValueError("block operator residual exceeds bound")
        object.__setattr__(self, "input_instruction_indices", inputs)
        object.__setattr__(self, "output_instruction_indices", outputs)
        object.__setattr__(self, "physical_qubits", pair)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_instruction_indices": list(self.input_instruction_indices),
            "output_instruction_indices": list(self.output_instruction_indices),
            "physical_qubits": list(self.physical_qubits),
            "baseline_cz_count": self.baseline_cz_count,
            "replacement_cz_count": self.replacement_cz_count,
            "equivalence": self.equivalence,
            "phase_real": self.phase_real,
            "phase_imag": self.phase_imag,
            "max_abs_residual": self.max_abs_residual,
        }

    @classmethod
    def from_dict(cls, payload: object) -> BlockRewriteEvidence:
        return cls(**_closed(payload, _REWRITE_FIELDS, "block rewrite"))


@dataclass(frozen=True)
class PhysicalInstructionEvidenceV4:
    """A final instruction owned by one unchanged native input or one block."""

    instruction_index: int
    native_input_instruction_index: int | None
    block_rewrite_index: int | None
    block_replacement_ordinal: int | None
    opcode: str
    physical_qubits: tuple[int, ...]
    layer: int
    predecessors: tuple[int, ...]
    dependency_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        _index(self.instruction_index, "instruction_index")
        _index(self.layer, "layer")
        for name in (
            "native_input_instruction_index",
            "block_rewrite_index",
            "block_replacement_ordinal",
        ):
            if getattr(self, name) is not None:
                _index(getattr(self, name), name)
        unchanged = self.native_input_instruction_index is not None
        if unchanged:
            if (
                self.block_rewrite_index is not None
                or self.block_replacement_ordinal is not None
            ):
                raise ValueError("unchanged output must have null block fields")
        elif self.block_rewrite_index is None or self.block_replacement_ordinal is None:
            raise ValueError("rewritten output requires block and ordinal")
        physical = _qubits(self.physical_qubits, "physical_qubits")
        _opcode(self.opcode, physical)
        predecessors = _indices(self.predecessors, "predecessors", _MAX_PREDECESSORS)
        if any(index >= self.instruction_index for index in predecessors):
            raise ValueError("predecessors must precede the instruction")
        kinds_list: list[str] = []
        for kind in _array(
            self.dependency_kinds, "dependency_kinds", len(_DEPENDENCY_KINDS)
        ):
            if not isinstance(kind, str) or kind not in _DEPENDENCY_KINDS:
                raise ValueError("dependency kinds must be sorted supported values")
            kinds_list.append(kind)
        kinds = tuple(kinds_list)
        if kinds != tuple(sorted(set(kinds))):
            raise ValueError("dependency kinds must be sorted supported values")
        if bool(predecessors) != bool(kinds):
            raise ValueError("physical dependency kinds are inconsistent")
        object.__setattr__(self, "physical_qubits", physical)
        object.__setattr__(self, "predecessors", predecessors)
        object.__setattr__(self, "dependency_kinds", kinds)

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction_index": self.instruction_index,
            "native_input_instruction_index": self.native_input_instruction_index,
            "block_rewrite_index": self.block_rewrite_index,
            "block_replacement_ordinal": self.block_replacement_ordinal,
            "opcode": self.opcode,
            "physical_qubits": list(self.physical_qubits),
            "layer": self.layer,
            "predecessors": list(self.predecessors),
            "dependency_kinds": list(self.dependency_kinds),
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhysicalInstructionEvidenceV4:
        return cls(**_closed(payload, _INSTRUCTION_FIELDS, "physical instruction"))


@dataclass(frozen=True)
class PhysicalPlanEvidenceV4:
    """Closed unallocated plan with exclusive many-to-many block lineage."""

    source_circuit_hash: str
    physical_circuit_hash: str
    target_snapshot_id: str
    topology_identity: str | None
    coupling: CouplingEvidence | None
    initial_logical_to_physical: tuple[int, ...]
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    mapping_transitions: tuple[MappingTransitionEvidence, ...]
    instructions: tuple[PhysicalInstructionEvidenceV4, ...]
    topology_legalization_identity: str | None
    native_gate_legalization_identity: str
    schedule_identity: str
    schedule_depth: int
    maximum_parallel_width: int
    critical_path: tuple[int, ...]
    native_input_circuit_hash: str
    native_input_instructions: tuple[NativeInputInstructionEvidence, ...]
    block_rewrites: tuple[BlockRewriteEvidence, ...]
    native_block_optimization_identity: str
    mapping_transition_space: str = "topology_program"
    plan_identity: str = ""

    def __post_init__(self) -> None:
        for name in (
            "source_circuit_hash",
            "physical_circuit_hash",
            "target_snapshot_id",
            "native_gate_legalization_identity",
            "schedule_identity",
            "native_input_circuit_hash",
            "native_block_optimization_identity",
        ):
            _sha256(getattr(self, name), name)
        if self.mapping_transition_space != "topology_program":
            raise ValueError("mapping_transition_space must be topology_program")
        for name, record_type in (
            ("mapping_transitions", MappingTransitionEvidence),
            ("instructions", PhysicalInstructionEvidenceV4),
            ("native_input_instructions", NativeInputInstructionEvidence),
            ("block_rewrites", BlockRewriteEvidence),
        ):
            records = _array(getattr(self, name), name)
            if any(not isinstance(item, record_type) for item in records):
                raise TypeError(f"{name} must contain typed evidence records")
            object.__setattr__(self, name, records)
        if not self.block_rewrites:
            raise ValueError("version 4 requires at least one block rewrite")
        self._validate_layout()
        self._validate_membership()
        self._validate_schedule()
        expected = hashlib.sha256(
            _canonical_bytes(self._plan_identity_payload())
        ).hexdigest()
        if self.plan_identity and self.plan_identity != expected:
            raise ValueError("plan_identity does not match physical plan evidence")
        object.__setattr__(self, "plan_identity", expected)
        if len(_canonical_bytes(self.to_dict())) > _MAX_BUNDLE_BYTES:
            raise ValueError("physical plan exceeds maximum UTF-8 bytes")

    def _validate_layout(self) -> None:
        for name in (
            "initial_logical_to_physical",
            "pre_restore_logical_to_physical",
            "final_logical_to_physical",
        ):
            _array(getattr(self, name), name)
            layout = (
                _indices(getattr(self, name), name)
                if name != "pre_restore_logical_to_physical"
                else _int_tuple(getattr(self, name), name)
            )
            object.__setattr__(self, name, layout)
        initial = self.initial_logical_to_physical
        final = self.final_logical_to_physical
        if not initial or initial != tuple(range(len(initial))) or final != initial:
            raise ValueError(
                "initial and final layouts must be the same dense identity"
            )
        if sorted(self.pre_restore_logical_to_physical) != list(initial):
            raise ValueError("pre-restore layout must be a dense permutation")
        _sha256(self.topology_identity, "topology_identity", optional=True)
        _sha256(
            self.topology_legalization_identity,
            "topology_legalization_identity",
            optional=True,
        )
        if self.coupling is None:
            if (
                self.topology_identity is not None
                or self.topology_legalization_identity is not None
                or self.mapping_transitions
            ):
                raise ValueError("absent topology cannot carry mapping evidence")
        elif not isinstance(self.coupling, CouplingEvidence):
            raise TypeError("version 4 requires undirected coupling evidence")
        elif (
            self.topology_identity != _coupling_identity(self.coupling)
            or self.topology_legalization_identity is None
            or self.coupling.n_wires < len(initial)
        ):
            raise ValueError("coupling and topology identities are inconsistent")
        replay_layout = list(initial)
        before_restore = initial
        restoring = False
        last_index = -1
        routing_groups: dict[int, list[NativeInputInstructionEvidence]] = {}
        for item in self.native_input_instructions:
            if item.origin in {"routing_swap", "routing_swap_decomposition"}:
                routing_groups.setdefault(item.topology_instruction_index, []).append(
                    item
                )
        for transition in self.mapping_transitions:
            if (
                transition.routed_instruction_index <= last_index
                or transition.routed_instruction_index not in routing_groups
            ):
                raise ValueError(
                    "mapping transition must name an ordered topology routing SWAP"
                )
            group = routing_groups[transition.routed_instruction_index]
            if any(
                item.source_instruction_index != transition.source_instruction_index
                or not set(item.physical_qubits) <= set(transition.physical_wires)
                for item in group
            ):
                raise ValueError("mapping transition ancestry or pair is inconsistent")
            last_index = transition.routed_instruction_index
            if transition.layout_before != tuple(replay_layout) or len(
                transition.layout_before
            ) != len(initial):
                raise ValueError("mapping transitions are not continuous")
            if any(wire >= len(initial) for wire in transition.physical_wires):
                raise ValueError("mapping transition uses an unallocated wire")
            if (
                self.coupling is None
                or tuple(sorted(transition.physical_wires)) not in self.coupling.edges
            ):
                raise ValueError("mapping transition violates coupling")
            if restoring and transition.phase != "final_restore":
                raise ValueError("forward movement cannot follow final restoration")
            if transition.phase == "final_restore" and not restoring:
                before_restore = tuple(replay_layout)
                restoring = True
            _swap_layout(replay_layout, transition.physical_wires)
            if transition.layout_after != tuple(replay_layout):
                raise ValueError("mapping transition result is inconsistent")
        if not restoring:
            before_restore = tuple(replay_layout)
        if (
            tuple(replay_layout) != final
            or before_restore != self.pre_restore_logical_to_physical
        ):
            raise ValueError("mapping replay does not match declared layouts")
        if set(routing_groups) != {
            item.routed_instruction_index for item in self.mapping_transitions
        }:
            raise ValueError(
                "routing native inputs require exactly their mapping transitions"
            )

    def _validate_membership(self) -> None:
        native = self.native_input_instructions
        output = self.instructions
        for index, native_item in enumerate(native):
            if native_item.instruction_index != index:
                raise ValueError("native input indices must be dense and ordered")
            if (
                self.coupling is not None
                and len(native_item.physical_qubits) == 2
                and tuple(sorted(native_item.physical_qubits))
                not in self.coupling.edges
            ):
                raise ValueError("native input violates coupling")
            if any(
                wire >= len(self.initial_logical_to_physical)
                for wire in (
                    *native_item.source_logical_qubits,
                    *native_item.physical_qubits,
                )
            ):
                raise ValueError("native input refers to an unallocated wire")
            if (
                native_item.topology_instruction_index >= _MAX_RECORDS
                or native_item.source_instruction_index >= _MAX_RECORDS
            ):
                raise ValueError("native ancestry exceeds record index limit")
            previous = native[index - 1] if index else None
            if (
                previous is not None
                and native_item.topology_instruction_index
                == previous.topology_instruction_index
            ):
                if (
                    native_item.native_replacement_ordinal
                    != previous.native_replacement_ordinal + 1
                    or native_item.source_instruction_index
                    != previous.source_instruction_index
                    or native_item.source_logical_qubits
                    != previous.source_logical_qubits
                    or native_item.origin != previous.origin
                ):
                    raise ValueError("native replacement ancestry is inconsistent")
            elif native_item.native_replacement_ordinal != 0 or (
                previous is not None
                and native_item.topology_instruction_index
                <= previous.topology_instruction_index
            ):
                raise ValueError(
                    "native topology groups must be ordered with dense ordinals"
                )
        membership_size = sum(
            len(block.input_instruction_indices) + len(block.output_instruction_indices)
            for block in self.block_rewrites
        )
        if membership_size > _MAX_RECORDS:
            raise ValueError("nested block memberships exceed record limit")
        used_inputs: set[int] = set()
        used_outputs: set[int] = set()
        for block_index, block in enumerate(self.block_rewrites):
            inputs, outputs = (
                block.input_instruction_indices,
                block.output_instruction_indices,
            )
            if any(index >= len(native) for index in inputs) or any(
                index >= len(output) for index in outputs
            ):
                raise ValueError("block membership index is outside its program")
            if used_inputs.intersection(inputs) or used_outputs.intersection(outputs):
                raise ValueError("block memberships must be disjoint")
            used_inputs.update(inputs)
            used_outputs.update(outputs)
            pair = set(block.physical_qubits)
            if any(
                native[index].opcode not in {"rx", "rz", "cz"}
                or not set(native[index].physical_qubits) <= pair
                for index in inputs
            ):
                raise ValueError(
                    "block inputs must be supported native operations on its pair"
                )
            members = set(inputs)
            if any(
                pair.intersection(native[index].physical_qubits)
                for index in range(inputs[0], inputs[-1] + 1)
                if index not in members
            ):
                raise ValueError("block crosses an intervening operation on its pair")
            if (
                sum(native[index].opcode == "cz" for index in inputs)
                != block.baseline_cz_count
            ):
                raise ValueError("block baseline CZ count is inconsistent")
            if (
                sum(output[index].opcode == "cz" for index in outputs)
                != block.replacement_cz_count
            ):
                raise ValueError("block replacement CZ count is inconsistent")
            for ordinal, index in enumerate(outputs):
                item = output[index]
                if (
                    item.block_rewrite_index != block_index
                    or item.block_replacement_ordinal != ordinal
                    or item.native_input_instruction_index is not None
                ):
                    raise ValueError("block output reference is inconsistent")
                if (
                    item.opcode not in {"rx", "rz", "cz"}
                    or not set(item.physical_qubits) <= pair
                ):
                    raise ValueError(
                        "block output must use the native basis on its pair"
                    )
        # Each output has a source position. Nondecreasing positions on every
        # touched qubit permit moving a block only across disjoint operations.
        last_source_position: dict[int, int] = {}
        for index, item in enumerate(output):
            if item.instruction_index != index:
                raise ValueError(
                    "physical instruction indices must be dense and ordered"
                )
            if any(
                wire >= len(self.initial_logical_to_physical)
                for wire in item.physical_qubits
            ):
                raise ValueError("physical instruction uses an unallocated wire")
            if (
                self.coupling is not None
                and len(item.physical_qubits) == 2
                and tuple(sorted(item.physical_qubits)) not in self.coupling.edges
            ):
                raise ValueError("physical instruction violates coupling")
            if item.native_input_instruction_index is not None:
                source_index = item.native_input_instruction_index
                if source_index >= len(native) or source_index in used_inputs:
                    raise ValueError(
                        "unchanged input membership must be exclusive and in range"
                    )
                original = native[source_index]
                if (item.opcode, item.physical_qubits) != (
                    original.opcode,
                    original.physical_qubits,
                ):
                    raise ValueError("unchanged instruction differs from native input")
                used_inputs.add(source_index)
                used_outputs.add(index)
                anchor = source_index
            else:
                if (
                    index not in used_outputs
                    or item.block_rewrite_index is None
                    or item.block_rewrite_index >= len(self.block_rewrites)
                ):
                    raise ValueError("rewritten output is not covered by its block")
                anchor = self.block_rewrites[
                    item.block_rewrite_index
                ].input_instruction_indices[0]
            for wire in item.physical_qubits:
                if last_source_position.get(wire, -1) > anchor:
                    raise ValueError("output reorders operations on a shared wire")
                last_source_position[wire] = anchor
        if used_inputs != set(range(len(native))) or used_outputs != set(
            range(len(output))
        ):
            raise ValueError("native input and output memberships must be complete")

    def _validate_schedule(self) -> None:
        instructions = self.instructions
        last_on_wire: dict[int, int] = {}
        for item in instructions:
            for predecessor in item.predecessors:
                if instructions[predecessor].layer >= item.layer:
                    raise ValueError("dependency must come from an earlier layer")
            expected_layer = 1 + max(
                (instructions[index].layer for index in item.predecessors), default=-1
            )
            if item.layer != expected_layer:
                raise ValueError("instruction layer does not match its dependencies")
            for wire in item.physical_qubits:
                previous = last_on_wire.get(wire)
                if previous is not None and previous not in item.predecessors:
                    raise ValueError("schedule omits a shared-wire dependency")
                last_on_wire[wire] = item.instruction_index
        depth = _index(self.schedule_depth, "schedule_depth")
        width = _index(self.maximum_parallel_width, "maximum_parallel_width")
        expected_depth = max((item.layer + 1 for item in instructions), default=0)
        if depth != expected_depth or width != max(
            Counter(item.layer for item in instructions).values(), default=0
        ):
            raise ValueError("physical plan schedule summary is inconsistent")
        path: list[int] = []
        if instructions:
            current = min(
                (item for item in instructions if item.layer == depth - 1),
                key=lambda item: item.instruction_index,
            )
            path.append(current.instruction_index)
            while current.predecessors:
                index = min(
                    current.predecessors,
                    key=lambda index: (-instructions[index].layer, index),
                )
                current = instructions[index]
                path.append(index)
        expected_path = tuple(reversed(path))
        if _int_tuple(self.critical_path, "critical_path") != expected_path:
            raise ValueError("physical plan critical path is inconsistent")
        object.__setattr__(self, "critical_path", expected_path)

    def _plan_identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("plan_identity")
        payload.pop("coupling")
        payload.update(
            {
                "schema": "flagquantum.physical_circuit_plan",
                "version": "4.0",
                "target_snapshot_id": self.target_snapshot_id,
                "coupling_n_wires": (
                    None if self.coupling is None else self.coupling.n_wires
                ),
                "coupling_edges": () if self.coupling is None else self.coupling.edges,
            }
        )
        return payload

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
            "native_input_circuit_hash": self.native_input_circuit_hash,
            "native_input_instructions": [
                item.to_dict() for item in self.native_input_instructions
            ],
            "block_rewrites": [item.to_dict() for item in self.block_rewrites],
            "native_block_optimization_identity": self.native_block_optimization_identity,
            "mapping_transition_space": self.mapping_transition_space,
        }

    @classmethod
    def from_dict(
        cls, payload: object, *, target_snapshot_id: str
    ) -> PhysicalPlanEvidenceV4:
        values = _closed(payload, _PLAN_FIELDS_V4, "physical plan evidence")
        coupling = values.pop("coupling")
        native = _array(
            values.pop("native_input_instructions"), "native_input_instructions"
        )
        blocks = _array(values.pop("block_rewrites"), "block_rewrites")
        outputs = _array(values.pop("instructions"), "instructions")
        transitions = _array(values.pop("mapping_transitions"), "mapping_transitions")
        membership_size = 0
        for block in blocks:
            record = _closed(block, _REWRITE_FIELDS, "block rewrite")
            membership_size += len(
                _array(record["input_instruction_indices"], "block inputs", 64)
            )
            membership_size += len(
                _array(record["output_instruction_indices"], "block outputs", 96)
            )
            if membership_size > _MAX_RECORDS:
                raise ValueError("nested block memberships exceed record limit")
        return cls(
            target_snapshot_id=target_snapshot_id,
            coupling=None if coupling is None else CouplingEvidence.from_dict(coupling),
            native_input_instructions=tuple(
                NativeInputInstructionEvidence.from_dict(item) for item in native
            ),
            block_rewrites=tuple(
                BlockRewriteEvidence.from_dict(item) for item in blocks
            ),
            instructions=tuple(
                PhysicalInstructionEvidenceV4.from_dict(item) for item in outputs
            ),
            mapping_transitions=tuple(
                MappingTransitionEvidence.from_dict(item) for item in transitions
            ),
            **values,
        )


@dataclass(frozen=True)
class CompilationEvidenceBundleV4:
    """Strict serialized evidence joining compilation inputs, plan, and output."""

    producer: str
    source: Mapping[str, str | None]
    target: Mapping[str, str]
    physical_plan: PhysicalPlanEvidenceV4
    output: Mapping[str, str]
    bundle_identity: str = ""
    version: str = "4.0"

    def __post_init__(self) -> None:
        if self.version != "4.0":
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
        if not isinstance(self.physical_plan, PhysicalPlanEvidenceV4):
            raise TypeError("physical_plan must be PhysicalPlanEvidenceV4")
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
    def from_dict(cls, payload: object) -> CompilationEvidenceBundleV4:
        values = _closed(payload, _TOP_LEVEL_FIELDS, "compilation evidence bundle")
        if values.pop("schema") != _SCHEMA:
            raise ValueError("invalid compilation evidence schema")
        target = _closed(values["target"], _TARGET_FIELDS, "compilation target")
        return cls(
            producer=values["producer"],
            version=values["version"],
            source=values["source"],
            target=target,
            physical_plan=PhysicalPlanEvidenceV4.from_dict(
                values["physical_plan"],
                target_snapshot_id=target["snapshot_id"],
            ),
            output=values["output"],
            bundle_identity=values["bundle_identity"],
        )
