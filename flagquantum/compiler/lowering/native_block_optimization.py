"""Retain and verify bounded collective rewrites of a native CZ baseline.

This private stage owns no routing decisions or serialized physical plan.
Annotations remain boundaries unless the owning router's retained result passes
replay. The target workflow connects this stage with v4 lineage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType

from ...core._compilation_evidence_v4 import BlockRewriteEvidence
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import MEASURE_OPCODE, RESET_OPCODE
from ...core.operator_schema import get_operator_schema
from ...core.target_capabilities import TargetCapabilitySnapshot
from ...errors import CompilationError
from ..optimization._cz_blocks import (
    MatrixReader,
    RejectionReason,
    _candidate,
    _collect_blocks,
    _equivalence_witness,
    _matrix,
    _numeric_matrix,
    _product,
)
from ..topology.routing import CouplingMap
from ..topology.topology_legalization import (
    TopologyLegalizationError,
    TopologyLegalizationResult,
    legalize_circuit_topology,
)
from .native_gate_legalization import (
    NativeGateLegalizationResult,
    _native_descriptors,
    _supports,
    legalize_native_gates,
)

_REJECTION_REASONS: tuple[RejectionReason, ...] = (
    "not_lower_cz_cost",
    "synthesis_unavailable",
    "residual_rejected",
    "instruction_growth_rejected",
)


class NativeBlockOptimizationError(CompilationError):
    """A retained native block rewrite fails its input or operator contract."""


@dataclass(frozen=True)
class NativeBlockOptimizationResult:
    """A native baseline and its collective rewrites, without flattened ancestry."""

    source_program: CircuitIR = field(repr=False)
    program: CircuitIR
    source_content_hash: str
    target_snapshot_id: str
    rewrites: tuple[BlockRewriteEvidence, ...]
    rejection_counts: Mapping[str, int]
    optimization_identity: str

    def __post_init__(self) -> None:
        if self.source_content_hash != self.source_program.content_hash:
            raise NativeBlockOptimizationError("native block source hash is stale")
        counts = dict(self.rejection_counts)
        if set(counts) != set(_REJECTION_REASONS) or any(
            type(value) is not int or value < 0 for value in counts.values()
        ):
            raise NativeBlockOptimizationError(
                "invalid native block rejection counters"
            )
        object.__setattr__(self, "rejection_counts", MappingProxyType(counts))


def _identity(
    source: CircuitIR,
    program: CircuitIR,
    target_snapshot_id: str,
    rewrites: tuple[BlockRewriteEvidence, ...],
    rejection_counts: Mapping[str, int],
) -> str:
    return _identity_from_hashes(
        source.content_hash,
        program.content_hash,
        target_snapshot_id,
        rewrites,
        rejection_counts,
    )


def _identity_from_hashes(
    source_content_hash: str,
    output_content_hash: str,
    target_snapshot_id: str,
    rewrites: tuple[BlockRewriteEvidence, ...],
    rejection_counts: Mapping[str, int],
) -> str:
    payload = {
        "source_content_hash": source_content_hash,
        "output_content_hash": output_content_hash,
        "target_snapshot_id": target_snapshot_id,
        "rewrites": [rewrite.to_dict() for rewrite in rewrites],
        "rejection_counts": dict(rejection_counts),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _admitted_target(
    snapshot: TargetCapabilitySnapshot, evaluated_at: datetime | None
) -> bool:
    descriptors = _native_descriptors(snapshot, evaluated_at=evaluated_at)
    entanglers = {
        opcode
        for opcode in descriptors
        if opcode not in {MEASURE_OPCODE, RESET_OPCODE}
        and ((schema := get_operator_schema(opcode)) is None or schema.arity != 1)
    }
    return entanglers == {"cz"} and all(
        _supports(descriptors, Instruction(opcode, (0,), {"theta": 0.0}))
        for opcode in ("rz", "rx")
    )


def _budget(max_added_operations: int) -> None:
    if type(max_added_operations) is not int:
        raise TypeError("max_added_operations must be an integer")
    if max_added_operations < 0:
        raise ValueError("max_added_operations must be non-negative")


def _safe_spectator(
    instruction: Instruction, matrix_reader: MatrixReader = _matrix
) -> bool:
    if len(instruction.wires) not in (1, 2):
        return False
    pair = (
        tuple(sorted(instruction.wires))
        if len(instruction.wires) == 2
        else (instruction.wires[0], instruction.wires[0] + 1)
    )
    return matrix_reader(instruction, (pair[0], pair[1])) is not None


def _native_matrix_reader(
    native: NativeGateLegalizationResult,
    *,
    topology: TopologyLegalizationResult | None,
    coupling_map: CouplingMap | None,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None,
    max_added_operations: int,
) -> MatrixReader:
    if topology is None:
        if coupling_map is not None:
            raise NativeBlockOptimizationError("coupling map has no retained routing")
    elif (
        coupling_map is None
        or topology.direction_semantics != "undirected"
        or topology.allocation_identity is not None
        or native.source_program is not topology.program
    ):
        raise NativeBlockOptimizationError(
            "native blocks require retained undirected routing"
        )
    elif coupling_map is not None:
        try:
            replay = legalize_circuit_topology(
                topology.source_program,
                coupling_map=coupling_map,
                snapshot=snapshot,
                strategy=topology.strategy,
                max_added_operations=max(
                    0,
                    len(topology.program.instructions)
                    - len(topology.source_program.instructions),
                ),
            )
        except TopologyLegalizationError as error:
            raise NativeBlockOptimizationError(
                "retained topology failed routing replay"
            ) from error
        # Compare all result fields without invoking tensor-valued IR equality.
        if (
            replay.program.content_hash != topology.program.content_hash
            or replace(
                topology, source_program=replay.source_program, program=replay.program
            )
            != replay
        ):
            raise NativeBlockOptimizationError(
                "retained topology failed routing replay"
            )
    _verify_native_baseline(native, snapshot, evaluated_at, max_added_operations)
    decompositions = {item.instruction_index: item for item in native.decompositions}
    trusted: set[int] = set()
    native_index = 0
    for index, instruction in enumerate(native.source_program.instructions):
        original = instruction
        if topology is not None:
            source_index = instruction.metadata["source_instruction_index"]
            original = topology.source_program.instructions[source_index]
        # Source operations carry layout_mapped, or routed in the per-gate
        # materializer. Inserted SWAPs use their source only as an audit anchor.
        inserted_swap = (
            topology is not None
            and "layout_mapped" not in instruction.metadata
            and "routed" not in instruction.metadata
        )
        decomposition = decompositions.get(index)
        count = 1 if decomposition is None else len(decomposition.replacement_opcodes)
        if inserted_swap or (not original.metadata and original.matrix is None):
            trusted.update(
                id(item)
                for item in native.program.instructions[
                    native_index : native_index + count
                ]
            )
        native_index += count

    def read(
        instruction: Instruction, pair: tuple[int, int]
    ) -> list[list[complex]] | None:
        if id(instruction) in trusted:
            return (
                _numeric_matrix(instruction, pair)
                if topology is not None
                else _matrix(instruction, pair)
            )
        return None

    return read


def optimize_native_blocks(
    native: NativeGateLegalizationResult,
    *,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None = None,
    max_added_operations: int = 256,
    topology: TopologyLegalizationResult | None = None,
    coupling_map: CouplingMap | None = None,
) -> NativeBlockOptimizationResult | None:
    """Propose verified CZ reductions while retaining the exact native baseline.

    A nonqualifying target has no stage. A qualifying target with no accepted
    rewrite retains the identical program instance. Neither case emits v4.
    """
    _budget(max_added_operations)
    if native.target_snapshot_id != snapshot.snapshot_id:
        raise NativeBlockOptimizationError("native baseline targets another snapshot")
    if not _admitted_target(snapshot, evaluated_at):
        return None
    source = native.program
    limit = len(native.source_program.instructions) + max_added_operations
    if len(source.instructions) > limit:
        raise NativeBlockOptimizationError("native baseline already exceeds budget")
    matrix_reader = _native_matrix_reader(
        native,
        topology=topology,
        coupling_map=coupling_map,
        snapshot=snapshot,
        evaluated_at=evaluated_at,
        max_added_operations=max_added_operations,
    )
    counts: dict[str, int] = dict.fromkeys(_REJECTION_REASONS, 0)
    replacements: dict[int, tuple[Instruction, ...]] = {}
    accepted: list[tuple[tuple[int, ...], tuple[int, int], complex, float]] = []
    output_size = len(source.instructions)

    def rejected(reason: RejectionReason) -> None:
        counts[reason] += 1

    for pair, positions in _collect_blocks(
        source,
        lambda index, pair: source.instructions[index].name in {"rx", "rz", "cz"}
        and matrix_reader(source.instructions[index], pair) is not None,
        barrier=lambda instruction: not _safe_spectator(instruction, matrix_reader),
        identity_reader=matrix_reader,
    ):
        block = [source.instructions[index] for index in positions]
        candidate = _candidate(
            block,
            pair,
            max_output_operations=len(block) + limit - output_size,
            on_rejection=rejected,
            matrix_reader=matrix_reader,
        )
        if candidate is None:
            continue
        before, after = _product(block, pair, matrix_reader=matrix_reader), _product(
            candidate, pair
        )
        witness = (
            None
            if before is None or after is None
            else _equivalence_witness(before, after)
        )
        if witness is None:
            raise NativeBlockOptimizationError(
                "accepted proposal lost its operator proof"
            )
        phase, residual = witness
        accepted.append((positions, pair, phase, residual))
        replacements[positions[0]] = candidate
        replacements.update((index, ()) for index in positions[1:])
        output_size += len(candidate) - len(block)

    output: list[Instruction] = []
    output_starts: dict[int, int] = {}
    for index, instruction in enumerate(source.instructions):
        output_starts[index] = len(output)
        output.extend(replacements.get(index, (instruction,)))
    rewrites = tuple(
        BlockRewriteEvidence(
            input_instruction_indices=positions,
            output_instruction_indices=tuple(
                range(
                    output_starts[positions[0]],
                    output_starts[positions[0]] + len(replacements[positions[0]]),
                )
            ),
            physical_qubits=pair,
            baseline_cz_count=sum(
                source.instructions[index].name == "cz" for index in positions
            ),
            replacement_cz_count=sum(
                item.name == "cz" for item in replacements[positions[0]]
            ),
            equivalence="unitary_up_to_global_phase",
            phase_real=phase.real,
            phase_imag=phase.imag,
            max_abs_residual=residual,
        )
        for positions, pair, phase, residual in sorted(accepted)
    )
    program = replace(source, instructions=tuple(output)) if rewrites else source
    # Share only this construction's hashes; the constructor and later verifier
    # still read the source again to detect mutable circuit data.
    source_content_hash = source.content_hash
    output_content_hash = (
        source_content_hash if program is source else program.content_hash
    )
    return NativeBlockOptimizationResult(
        source_program=source,
        program=program,
        source_content_hash=source_content_hash,
        target_snapshot_id=snapshot.snapshot_id,
        rewrites=rewrites,
        rejection_counts=counts,
        optimization_identity=_identity_from_hashes(
            source_content_hash,
            output_content_hash,
            snapshot.snapshot_id,
            rewrites,
            counts,
        ),
    )


def verify_native_block_optimization(
    result: NativeBlockOptimizationResult,
    native: NativeGateLegalizationResult,
    *,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None = None,
    max_added_operations: int = 256,
    topology: TopologyLegalizationResult | None = None,
    coupling_map: CouplingMap | None = None,
) -> None:
    """Recompute operators and ancestry; a recorded residual is never the proof."""
    _budget(max_added_operations)
    if result.source_program is not native.program:
        raise NativeBlockOptimizationError(
            "native block stage does not retain the baseline instance"
        )
    if (
        result.target_snapshot_id != snapshot.snapshot_id
        or native.target_snapshot_id != snapshot.snapshot_id
    ):
        raise NativeBlockOptimizationError(
            "native block stage targets another snapshot"
        )
    if not _admitted_target(snapshot, evaluated_at):
        raise NativeBlockOptimizationError("target does not admit native CZ blocks")
    matrix_reader = _native_matrix_reader(
        native,
        topology=topology,
        coupling_map=coupling_map,
        snapshot=snapshot,
        evaluated_at=evaluated_at,
        max_added_operations=max_added_operations,
    )
    if result.optimization_identity != _identity(
        result.source_program,
        result.program,
        result.target_snapshot_id,
        result.rewrites,
        result.rejection_counts,
    ):
        raise NativeBlockOptimizationError(
            "native block optimization identity is stale"
        )
    source = result.source_program
    if (
        replace(result.program, instructions=source.instructions).content_hash
        != source.content_hash
    ):
        raise NativeBlockOptimizationError(
            "native block stage changed noninstruction semantics"
        )
    if (
        len(result.program.instructions)
        > len(native.source_program.instructions) + max_added_operations
    ):
        raise NativeBlockOptimizationError(
            "native block output exceeds combined budget"
        )
    _verify_lineage(result, matrix_reader)


def _verify_native_baseline(
    native: NativeGateLegalizationResult,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None,
    max_added_operations: int,
) -> None:
    baseline = legalize_native_gates(
        native.source_program,
        snapshot=snapshot,
        evaluated_at=evaluated_at,
        max_added_operations=max_added_operations,
    )
    if (
        baseline.legalization_identity != native.legalization_identity
        or baseline.program.content_hash != native.program.content_hash
        or baseline.decompositions != native.decompositions
        or baseline.native_opcodes != native.native_opcodes
        or native.source_content_hash != native.source_program.content_hash
    ):
        raise NativeBlockOptimizationError("retained native baseline failed replay")


def _verify_rewrite_operators(
    rewrite: BlockRewriteEvidence,
    source: CircuitIR,
    program: CircuitIR,
    matrix_reader: MatrixReader,
) -> None:
    block = [source.instructions[index] for index in rewrite.input_instruction_indices]
    output = [
        program.instructions[index] for index in rewrite.output_instruction_indices
    ]
    if any(item.name not in {"rx", "rz", "cz"} for item in (*block, *output)):
        raise NativeBlockOptimizationError("native block is outside the admitted basis")
    before, after = _product(
        block, rewrite.physical_qubits, matrix_reader=matrix_reader
    ), _product(output, rewrite.physical_qubits)
    witness = (
        None if before is None or after is None else _equivalence_witness(before, after)
    )
    if witness is None:
        raise NativeBlockOptimizationError("native block operator equivalence failed")
    phase, residual = witness
    if (
        abs(phase - complex(rewrite.phase_real, rewrite.phase_imag)) > 1e-11
        or abs(residual - rewrite.max_abs_residual) > 1e-11
    ):
        raise NativeBlockOptimizationError(
            "native block witness does not describe its operators"
        )
    if (
        sum(item.name == "cz" for item in block) != rewrite.baseline_cz_count
        or sum(item.name == "cz" for item in output) != rewrite.replacement_cz_count
    ):
        raise NativeBlockOptimizationError(
            "native block CZ counts do not match its operators"
        )


def _verify_lineage(
    result: NativeBlockOptimizationResult, matrix_reader: MatrixReader
) -> None:
    source = result.source_program
    memberships: dict[int, BlockRewriteEvidence] = {}
    for rewrite in result.rewrites:
        positions = rewrite.input_instruction_indices
        if any(
            index >= len(source.instructions) or index in memberships
            for index in positions
        ):
            raise NativeBlockOptimizationError(
                "invalid or overlapping native block membership"
            )
        if any(
            index >= len(result.program.instructions)
            for index in rewrite.output_instruction_indices
        ):
            raise NativeBlockOptimizationError(
                "native block output index is out of range"
            )
        memberships.update((index, rewrite) for index in positions)
        _verify_rewrite_operators(rewrite, source, result.program, matrix_reader)
    cursor = 0
    last_touch: dict[int, int] = {}
    last_barrier = -1
    previous_member: dict[int, int] = {}
    for index, instruction in enumerate(source.instructions):
        member_rewrite = memberships.get(index)
        if member_rewrite is not None:
            start = member_rewrite.input_instruction_indices[0]
            previous = previous_member.get(start)
            if previous is not None and (
                last_barrier > previous
                or any(
                    last_touch.get(qubit, -1) > previous
                    for qubit in member_rewrite.physical_qubits
                )
            ):
                raise NativeBlockOptimizationError(
                    "native block crosses an ordering dependency"
                )
            previous_member[start] = index
            if index == member_rewrite.input_instruction_indices[0]:
                expected = tuple(
                    range(
                        cursor, cursor + len(member_rewrite.output_instruction_indices)
                    )
                )
                if expected != member_rewrite.output_instruction_indices:
                    raise NativeBlockOptimizationError(
                        "native block was not emitted at its first input"
                    )
                cursor += len(expected)
            last_touch.update((qubit, index) for qubit in instruction.wires)
            continue
        if not _safe_spectator(instruction, matrix_reader):
            last_barrier = index
        last_touch.update((qubit, index) for qubit in instruction.wires)
        if (
            cursor >= len(result.program.instructions)
            or result.program.instructions[cursor] is not instruction
        ):
            raise NativeBlockOptimizationError(
                "unchanged native instruction lost its identity"
            )
        cursor += 1
    if cursor != len(result.program.instructions):
        raise NativeBlockOptimizationError(
            "native block output contains unowned instructions"
        )
