"""Private statevector compilation and tensor operations."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import TypeAlias

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ..gate_matrix import (
    gate_matrix as _gate_matrix,
)
from ..gate_matrix import (
    parameter_tensor as _parameter_tensor,
)
from ..matrices import GATE_MAT_DICT
from ..numerics.complex_arithmetic import complex_mul
from .diagonal_cpu import (
    _apply_cross_wire_diagonal_cpu as _apply_cross_wire_diagonal_cpu,
)
from .diagonal_cpu import (
    _apply_disjoint_diagonal_regions_cpu as _apply_disjoint_diagonal_regions_cpu,
)

_STATEVECTOR_LAYOUT_CACHE: dict[
    tuple[int, tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]
] = {}


def clear_statevector_layout_cache() -> None:
    """Clear the cached wire-layout computations for statevector gates."""

    _STATEVECTOR_LAYOUT_CACHE.clear()


# A CX sequence acts on basis indices as an affine map over GF(2), so the whole
# sequence can be applied as one gather instead of one pass per gate. The table
# costs one int32 per amplitude, which is half the size of a complex64 state, and
# it is addressed by its content rather than by a Circuit, so it cannot go stale
# when a Circuit is mutated. The budget bounds what distinct sequences may keep
# resident; the entries are dropped in insertion order once it is exceeded.
_CX_SEQUENCE_INDEX_CACHE_BYTES = 128 * 1024 * 1024
_CX_SEQUENCE_INDEX_CACHE: dict[
    tuple[int, tuple[int, ...], tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}


def clear_statevector_cx_index_cache() -> None:
    """Clear the cached CX-sequence permutation tables."""

    _CX_SEQUENCE_INDEX_CACHE.clear()


def _cx_sequence_index_cache_bytes() -> int:
    return sum(
        int(table.numel()) * int(table.element_size())
        for table in _CX_SEQUENCE_INDEX_CACHE.values()
    )


def _store_cx_sequence_index(
    key: tuple[int, tuple[int, ...], tuple[int, ...], str, torch.dtype],
    table: torch.Tensor,
) -> torch.Tensor:
    _CX_SEQUENCE_INDEX_CACHE[key] = table
    while (
        len(_CX_SEQUENCE_INDEX_CACHE) > 1
        and _cx_sequence_index_cache_bytes() > _CX_SEQUENCE_INDEX_CACHE_BYTES
    ):
        oldest = next(iter(_CX_SEQUENCE_INDEX_CACHE))
        del _CX_SEQUENCE_INDEX_CACHE[oldest]
    return table


_DIAGONAL_STATEVECTOR_GATES = frozenset(
    {"z", "s", "sdg", "t", "tdg", "rz", "p", "phase", "u1", "cz", "cphase", "rzz"}
)


def _diagonal_region(instructions: Sequence[Instruction]) -> bool:
    """Whether a whole fused region provably applies a diagonal matrix.

    A product of diagonal matrices is diagonal, so it is enough that every
    member is one, and a member is only known to be one when it is a named
    diagonal gate carrying no matrix of its own. ``gate_matrix`` returns a
    supplied ``matrix`` in preference to the gate the name denotes, so for an
    instruction that carries one the name says nothing about the operator and
    the region is called non-diagonal. That is the conservative answer and the
    one the caller needs, because this decides which kernel runs.

    The classification is a property of the program, so it is made once here
    rather than per execution: a region cannot become diagonal later.
    """

    return bool(instructions) and all(
        instruction.matrix is None
        and canonical_opcode(instruction.name) in _DIAGONAL_STATEVECTOR_GATES
        for instruction in instructions
    )


@dataclass(frozen=True)
class _StatevectorGateStep:
    instruction: Instruction
    layout: tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class _StatevectorRXRZLoopStep:
    wire: int
    pairs: tuple[tuple[Instruction, Instruction], ...]


@dataclass(frozen=True)
class _StatevectorFusedGateStep:
    instructions: tuple[Instruction, ...]
    wires: tuple[int, ...]
    layout: tuple[tuple[int, ...], tuple[int, ...]]
    dependency_reordered: bool = False
    # Decided when the region is formed, not when it runs, because which kernel
    # the region may take is a property of the program.
    diagonal: bool = False


@dataclass(frozen=True)
class _StatevectorCrossWireDiagonalStep:
    regions: tuple[_StatevectorGateStep | _StatevectorFusedGateStep, ...]


@dataclass(frozen=True)
class _StatevectorCXSequenceStep:
    controls: tuple[int, ...]
    targets: tuple[int, ...]


_StatevectorProgramStep: TypeAlias = (
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorFusedGateStep
    | _StatevectorCrossWireDiagonalStep
    | _StatevectorCXSequenceStep
)


def _instruction_matrix(
    instruction: Instruction, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    constant_parameters = instruction.matrix is None and all(
        isinstance(value, Number) for value in instruction.params.values()
    )
    construction_device = torch.device("cpu") if constant_parameters else device
    matrix = _gate_matrix(
        instruction,
        bsz=1,
        device=construction_device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError(
                "Local distributed simulator currently expects scalar gate parameters."
            )
        matrix = matrix[0]
    return matrix


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if default:
        return value not in {"0", "false", "off", "no"}
    return value in {"1", "true", "on", "yes"}


def _triton_single_qubit_loop_enabled() -> bool:
    return _environment_flag("FQ_TRITON_SINGLE_QUBIT_LOOP", default=True)


def _triton_ry_rz_pair_enabled() -> bool:
    return _environment_flag("FQ_TRITON_RY_RZ_PAIR", default=True)


def _triton_single_qubit_matrix_enabled() -> bool:
    return _environment_flag("FQ_TRITON_SINGLE_QUBIT_MATRIX", default=True)


def _triton_parameterized_single_qubit_matrix_enabled() -> bool:
    return _environment_flag(
        "FQ_TRITON_PARAMETERIZED_SINGLE_QUBIT_MATRIX", default=False
    )


# Sequence lengths at or above this one apply as a single gather. The table has
# to be built before it can be used, and measured against the per-gate loop on
# this host the build costs about as much as six to eight gates, so a shorter
# sequence is left on the loop. The threshold is a floor for the *unreused* case:
# a region executed more than once finds the table already cached.
_CX_SEQUENCE_GATHER_MINIMUM_LENGTH = 8


def _cpu_cx_sequence_gather_enabled() -> bool:
    """Whether a CPU CX sequence may be applied as one gather.

    On by default, like the other statevector fast-path switches. A gather is an
    exact permutation of amplitudes, so it cannot move a result even in the last
    bit; the switch exists to restore the per-gate loop if the table's memory or
    its build time is unwanted.
    """

    return _environment_flag("FQ_CPU_CX_SEQUENCE_GATHER", default=True)


def _cpu_single_wire_elementwise_enabled() -> bool:
    """Whether a CPU one-wire gate is applied by combining the wire's two halves.

    Off by default, unlike the other statevector fast-path switches, because this
    kernel is *not* bitwise equal to the ``bmm`` it replaces. ``bmm`` on a 2x2
    complex matrix accumulates the real and imaginary parts as separate sums over
    the two terms, while this kernel forms each complex product first and then
    adds, so the last bit can differ. Measured across both dtypes and four wires
    per width, the difference is zero up to seven wires - below that size ``bmm``
    takes the same path - and below 0.1 ulp of the real dtype above it: 1.1e-8 on
    a normalized complex64 state at eight wires, 3.3e-10 at twenty, and 6.9e-18
    on complex128 at twelve. That is small but it is not nothing, and the tests
    compare states with ``torch.equal``, so the switch stays opt-in rather than
    changing what the default path returns.
    """

    return _environment_flag("FQ_CPU_SINGLE_WIRE_ELEMENTWISE", default=False)


def _cpu_cross_wire_diagonal_fusion_enabled() -> bool:
    """Whether CPU wire-disjoint diagonal regions may share a statevector pass."""

    return _environment_flag("FQ_CPU_CROSS_WIRE_DIAGONAL_FUSION", default=True)


def _compile_statevector_program(
    instructions: Sequence[Instruction],
    n_wires: int,
    *,
    enable_triton_loop: bool,
    enable_cpu_cross_wire_diagonal: bool = False,
) -> tuple[_StatevectorProgramStep, ...]:
    program = _fuse_rx_rz_loops(
        instructions,
        n_wires,
        enable_triton_loop=enable_triton_loop,
    )
    fused_program = _fuse_gate_sequences(program)
    if enable_cpu_cross_wire_diagonal:
        return tuple(
            _fuse_cx_sequences(_fuse_cross_wire_diagonal_regions(fused_program))
        )
    return tuple(_fuse_cx_sequences(fused_program))


def _fuse_rx_rz_loops(
    instructions: Sequence[Instruction],
    n_wires: int,
    *,
    enable_triton_loop: bool,
) -> list[_StatevectorGateStep | _StatevectorRXRZLoopStep]:
    program: list[_StatevectorGateStep | _StatevectorRXRZLoopStep] = []
    index = 0
    while index < len(instructions):
        start = index
        segment: list[Instruction] = []
        while index < len(instructions):
            instruction = instructions[index]
            if (
                instruction.matrix is not None
                or instruction.name not in {"rx", "rz"}
                or len(instruction.wires) != 1
            ):
                break
            segment.append(instruction)
            index += 1
        fused = False
        if enable_triton_loop and segment:
            by_wire: dict[int, list[Instruction]] = {}
            for instruction in segment:
                wire = instruction.wires[0]
                if wire not in by_wire:
                    by_wire[wire] = []
                by_wire[wire].append(instruction)
            valid = all(
                len(items) >= 4
                and len(items) % 2 == 0
                and all(
                    items[offset].name == "rx" and items[offset + 1].name == "rz"
                    for offset in range(0, len(items), 2)
                )
                for items in by_wire.values()
            )
            if valid:
                for wire, items in by_wire.items():
                    program.append(
                        _StatevectorRXRZLoopStep(
                            wire,
                            tuple(
                                (items[offset], items[offset + 1])
                                for offset in range(0, len(items), 2)
                            ),
                        )
                    )
                fused = True
        if not fused:
            program.extend(
                _StatevectorGateStep(
                    instruction,
                    _statevector_layout(n_wires, instruction.wires),
                )
                for instruction in segment
            )
        if index == start:
            instruction = instructions[index]
            program.append(
                _StatevectorGateStep(
                    instruction,
                    _statevector_layout(n_wires, instruction.wires),
                )
            )
            index += 1
    return program


def _fuse_gate_sequences(
    program: Sequence[_StatevectorGateStep | _StatevectorRXRZLoopStep],
) -> list[_StatevectorGateStep | _StatevectorRXRZLoopStep | _StatevectorFusedGateStep]:
    fused_program: list[
        _StatevectorGateStep | _StatevectorRXRZLoopStep | _StatevectorFusedGateStep
    ] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not isinstance(step, _StatevectorGateStep):
            fused_program.append(step)
            index += 1
            continue
        if len(step.instruction.wires) == 1:
            cursor = index
            gate_steps_by_wire: dict[int, list[tuple[int, _StatevectorGateStep]]] = {}
            while cursor < len(program):
                candidate = program[cursor]
                if not (
                    isinstance(candidate, _StatevectorGateStep)
                    and len(candidate.instruction.wires) == 1
                ):
                    break
                wire = candidate.instruction.wires[0]
                gate_steps_by_wire.setdefault(wire, []).append(
                    (cursor - index, candidate)
                )
                cursor += 1
            for wire, positioned_steps in gate_steps_by_wire.items():
                if len(positioned_steps) == 1:
                    fused_program.append(positioned_steps[0][1])
                else:
                    positions = tuple(position for position, _ in positioned_steps)
                    instructions = tuple(
                        item.instruction for _, item in positioned_steps
                    )
                    fused_program.append(
                        _StatevectorFusedGateStep(
                            instructions=instructions,
                            wires=(wire,),
                            layout=positioned_steps[0][1].layout,
                            dependency_reordered=any(
                                right != left + 1
                                for left, right in zip(
                                    positions, positions[1:], strict=False
                                )
                            ),
                            diagonal=_diagonal_region(instructions),
                        )
                    )
            index = cursor
            continue
        group = [step]
        cursor = index + 1
        while cursor < len(program):
            candidate = program[cursor]
            if (
                not isinstance(candidate, _StatevectorGateStep)
                or candidate.instruction.wires != step.instruction.wires
            ):
                break
            group.append(candidate)
            cursor += 1
        if len(group) == 1:
            fused_program.append(step)
        else:
            instructions = tuple(item.instruction for item in group)
            fused_program.append(
                _StatevectorFusedGateStep(
                    instructions=instructions,
                    wires=tuple(step.instruction.wires),
                    layout=step.layout,
                    diagonal=_diagonal_region(instructions),
                )
            )
        index = cursor
    return fused_program


def _small_diagonal_region(
    step: _StatevectorGateStep | _StatevectorRXRZLoopStep | _StatevectorFusedGateStep,
) -> _StatevectorGateStep | _StatevectorFusedGateStep | None:
    if isinstance(step, _StatevectorGateStep):
        if len(step.instruction.wires) in {1, 2} and _diagonal_region(
            (step.instruction,)
        ):
            return step
        return None
    if (
        isinstance(step, _StatevectorFusedGateStep)
        and len(step.wires) in {1, 2}
        and step.diagonal
    ):
        return step
    return None


def _region_wires(
    region: _StatevectorGateStep | _StatevectorFusedGateStep,
) -> tuple[int, ...]:
    if isinstance(region, _StatevectorGateStep):
        return tuple(region.instruction.wires)
    return region.wires


def _disjoint_diagonal_groups(
    regions: Sequence[_StatevectorGateStep | _StatevectorFusedGateStep],
) -> list[list[_StatevectorGateStep | _StatevectorFusedGateStep]]:
    """Partition commuting diagonal regions into wire-disjoint matchings."""

    groups: list[list[_StatevectorGateStep | _StatevectorFusedGateStep]] = []
    used_wires: list[set[int]] = []
    for region in regions:
        wires = set(_region_wires(region))
        for group, occupied in zip(groups, used_wires, strict=True):
            if wires.isdisjoint(occupied):
                group.append(region)
                occupied.update(wires)
                break
        else:
            groups.append([region])
            used_wires.append(wires)
    return groups


def _fuse_cross_wire_diagonal_regions(
    program: Sequence[
        _StatevectorGateStep | _StatevectorRXRZLoopStep | _StatevectorFusedGateStep
    ],
) -> list[
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorFusedGateStep
    | _StatevectorCrossWireDiagonalStep
]:
    """Combine adjacent small diagonal regions into wire-disjoint matchings."""

    optimized: list[
        _StatevectorGateStep
        | _StatevectorRXRZLoopStep
        | _StatevectorFusedGateStep
        | _StatevectorCrossWireDiagonalStep
    ] = []
    index = 0
    while index < len(program):
        first = _small_diagonal_region(program[index])
        if first is None:
            optimized.append(program[index])
            index += 1
            continue
        regions = [first]
        cursor = index + 1
        while cursor < len(program):
            candidate = _small_diagonal_region(program[cursor])
            if candidate is None:
                break
            regions.append(candidate)
            cursor += 1
        for group in _disjoint_diagonal_groups(regions):
            if len(group) >= 2:
                optimized.append(_StatevectorCrossWireDiagonalStep(tuple(group)))
            else:
                optimized.append(group[0])
        index = cursor
    return optimized


def _fuse_cx_sequences(
    fused_program: Sequence[
        _StatevectorGateStep
        | _StatevectorRXRZLoopStep
        | _StatevectorFusedGateStep
        | _StatevectorCrossWireDiagonalStep
    ],
) -> list[_StatevectorProgramStep]:
    optimized_program: list[_StatevectorProgramStep] = []
    index = 0
    while index < len(fused_program):
        fused_step = fused_program[index]
        if not (
            isinstance(fused_step, _StatevectorGateStep)
            and canonical_opcode(fused_step.instruction.name) == "cx"
        ):
            optimized_program.append(fused_step)
            index += 1
            continue
        cx_steps = [fused_step]
        cursor = index + 1
        while cursor < len(fused_program):
            fused_candidate = fused_program[cursor]
            if not (
                isinstance(fused_candidate, _StatevectorGateStep)
                and canonical_opcode(fused_candidate.instruction.name) == "cx"
            ):
                break
            cx_steps.append(fused_candidate)
            cursor += 1
        if len(cx_steps) >= 2:
            optimized_program.append(
                _StatevectorCXSequenceStep(
                    controls=tuple(item.instruction.wires[0] for item in cx_steps),
                    targets=tuple(item.instruction.wires[1] for item in cx_steps),
                )
            )
        else:
            optimized_program.extend(cx_steps)
        index = cursor
    return optimized_program


def _statevector_layout(
    n_wires: int, wires: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    normalized_wires = tuple(wires)
    layout_key = (int(n_wires), normalized_wires)
    layout = _STATEVECTOR_LAYOUT_CACHE.get(layout_key)
    if layout is None:
        rest_wires = tuple(
            wire for wire in range(n_wires) if wire not in normalized_wires
        )
        perm = (
            (0,)
            + tuple(wire + 1 for wire in normalized_wires)
            + tuple(wire + 1 for wire in rest_wires)
        )
        inverse = [0] * len(perm)
        for i, axis in enumerate(perm):
            inverse[axis] = i
        layout = (perm, tuple(inverse))
        _STATEVECTOR_LAYOUT_CACHE[layout_key] = layout
    return layout


def _apply_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    wires = tuple(wires)
    if state.device.type == "cpu" and state.is_contiguous() and len(wires) == 1:
        return _apply_single_qubit_matrix_cpu(
            state,
            matrix,
            wire=wires[0],
            n_wires=n_wires,
        )
    if (
        state.device.type == "cpu"
        and state.is_contiguous()
        and wires == (int(n_wires) - 1, int(n_wires) - 2)
    ):
        return _apply_reversed_trailing_two_qubit_matrix_cpu(state, matrix)
    return _apply_matrix_layout(state, matrix, wires, n_wires, layout=layout)


def _apply_single_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Apply one CPU gate by visiting contiguous amplitude pairs directly."""

    if not 0 <= int(wire) < int(n_wires):
        raise ValueError("wire is outside the statevector")
    bsz = state.shape[0]
    stride = 1 << (int(n_wires) - int(wire) - 1)
    paired = state.reshape(bsz, -1, 2, stride)
    zero = paired[:, :, 0, :]
    one = paired[:, :, 1, :]
    matrix = matrix.to(device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(bsz, -1, -1)
    elif matrix.ndim == 3 and matrix.shape[0] == 1 and bsz != 1:
        matrix = matrix.expand(bsz, -1, -1)
    if matrix.shape != (bsz, 2, 2):
        raise ValueError("single-qubit matrix must have shape [2, 2] or [batch, 2, 2]")
    coefficient_shape = (bsz, 1, 1)
    out_zero = zero * matrix[:, 0, 0].reshape(coefficient_shape)
    out_zero = out_zero + one * matrix[:, 0, 1].reshape(coefficient_shape)
    out_one = zero * matrix[:, 1, 0].reshape(coefficient_shape)
    out_one = out_one + one * matrix[:, 1, 1].reshape(coefficient_shape)
    return torch.stack((out_zero, out_one), dim=2).reshape(state.shape)


def _apply_reversed_trailing_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply a CPU gate whose ordered wires are the two trailing state bits.

    Four amplitudes for those wires are already contiguous. Reordering the
    small gate matrix into memory-bit order therefore avoids permuting and
    materializing the full statevector around the matrix multiplication.
    """

    bsz = state.shape[0]
    matrix = matrix.to(device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(bsz, -1, -1)
    elif matrix.ndim == 3 and matrix.shape[0] == 1 and bsz != 1:
        matrix = matrix.expand(bsz, -1, -1)
    if matrix.shape != (bsz, 4, 4):
        raise ValueError("two-qubit matrix must have shape [4, 4] or [batch, 4, 4]")

    reordered = (
        matrix.reshape(bsz, 2, 2, 2, 2).permute(0, 2, 1, 4, 3).reshape(bsz, 1, 4, 4)
    )
    blocks = state.reshape(bsz, -1, 4, 1)
    return torch.matmul(reordered, blocks).reshape(state.shape)


def _apply_matrix_layout(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Apply a gate through the general layout-and-batched-matmul path."""

    wires = tuple(wires)
    bsz = state.shape[0]
    k = len(wires)
    dim = 2**k
    if layout is None:
        layout = _statevector_layout(n_wires, wires)
    perm, inv_perm = layout

    tensor = state.reshape((bsz,) + (2,) * n_wires).permute(perm)
    flat = tensor.reshape(bsz, dim, -1)
    if matrix.dtype != state.dtype or matrix.device != state.device:
        matrix = matrix.to(device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        matrix = matrix.expand(bsz, -1, -1)
    out = torch.bmm(matrix, flat)
    return out.reshape((bsz,) + (2,) * n_wires).permute(inv_perm).reshape(bsz, -1)


def _apply_single_wire_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Apply a one-wire gate by combining the two halves of that wire's axis.

    ``state`` is contiguous and wire ``w`` carries the bit of weight
    ``2 ** (n_wires - 1 - w)``, so ``reshape(bsz, 2 ** w, 2, 2 ** (n_wires - 1 - w))``
    is a view whose size-2 axis *is* that wire. The gate is then two scaled sums
    over that axis, which needs neither of the two layout permutations
    ``_apply_matrix`` performs and only one output allocation.

    This stands in for ``_apply_matrix`` and not for ``_apply_diagonal_matrix``:
    the two shipped kernels disagree with each other in the last bit, so a gate
    that would have been dispatched to the diagonal kernel keeps it. Even against
    ``_apply_matrix`` the agreement is not bitwise, for the reason recorded on
    ``_cpu_single_wire_elementwise_enabled``.
    """

    bsz = state.shape[0]
    wire = int(wire)
    if matrix.dtype != state.dtype or matrix.device != state.device:
        matrix = matrix.to(device=state.device, dtype=state.dtype)
    tensor = state.reshape(bsz, 2**wire, 2, 2 ** (n_wires - 1 - wire))
    low = tensor[:, :, 0, :]
    high = tensor[:, :, 1, :]
    if matrix.ndim == 2:
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
    else:
        entries = matrix.reshape(-1, 2, 2)
        if entries.shape[0] == 1:
            entries = entries.expand(bsz, -1, -1)
        a = entries[:, 0, 0].reshape(bsz, 1, 1)
        b = entries[:, 0, 1].reshape(bsz, 1, 1)
        c = entries[:, 1, 0].reshape(bsz, 1, 1)
        d = entries[:, 1, 1].reshape(bsz, 1, 1)
    combined = torch.stack((a * low + b * high, c * low + d * high), dim=2)
    return combined.reshape(bsz, -1)


def _apply_diagonal_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Apply a known diagonal gate without launching a dense batched matmul."""

    wires = tuple(wires)
    bsz = state.shape[0]
    dim = 2 ** len(wires)
    if state.device.type == "cpu" and state.is_contiguous():
        matrix = matrix.to(device=state.device, dtype=state.dtype)
        diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
        if diagonal.ndim == 1:
            diagonal = diagonal.unsqueeze(0).expand(bsz, -1)
        elif diagonal.shape[0] == 1 and bsz != 1:
            diagonal = diagonal.expand(bsz, -1)
        if diagonal.shape != (bsz, dim):
            raise ValueError(
                "diagonal matrix must have shape [dim, dim] or [batch, dim, dim]"
            )
        ordered_wires = tuple(sorted(wires))
        wire_order = tuple(wires.index(wire) for wire in ordered_wires)
        factors = diagonal.reshape((bsz,) + (2,) * len(wires)).permute(
            (0,) + tuple(index + 1 for index in wire_order)
        )
        factor_shape = [bsz] + [1] * n_wires
        for wire in ordered_wires:
            factor_shape[wire + 1] = 2
        tensor = state.reshape((bsz,) + (2,) * n_wires)
        return (tensor * factors.reshape(factor_shape)).reshape(state.shape)
    if layout is None:
        layout = _statevector_layout(n_wires, wires)
    perm, inv_perm = layout
    tensor = state.reshape((bsz,) + (2,) * n_wires).permute(perm)
    flat = tensor.reshape(bsz, dim, -1)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    if diagonal.ndim == 1:
        diagonal = diagonal.expand(bsz, -1)
    out = complex_mul(flat, diagonal.unsqueeze(-1))
    return out.reshape((bsz,) + (2,) * n_wires).permute(inv_perm).reshape(bsz, -1)


def _apply_cx_permutation(
    state: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply CNOT by flipping the target axis only in the control-one slice."""

    control, target = (int(wire) for wire in wires)
    tensor = state.reshape((state.shape[0],) + (2,) * n_wires)
    control_axis = control + 1
    target_axis = target + 1
    zero, one = tensor.unbind(dim=control_axis)
    if target_axis > control_axis:
        target_axis -= 1
    one = torch.flip(one, dims=(target_axis,))
    return torch.stack((zero, one), dim=control_axis).reshape(state.shape)


def _cx_sequence_index_dtype(n_wires: int) -> torch.dtype:
    """Index dtype for a permutation table over ``2 ** n_wires`` amplitudes.

    ``index_select`` takes int32, and an int32 index addresses every amplitude up
    to 31 wires, so the wider dtype is only needed past that point - where the
    state itself would already be 32 GiB per batch row.
    """

    return torch.int32 if n_wires <= 31 else torch.int64


def _cx_sequence_permutation_index(
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the gather table for one CX sequence.

    ``index_select(state, 1, table)`` sends ``state[:, source]`` to position
    ``destination`` where ``table[destination] == source``, so the table is the
    *inverse* of the map the sequence applies. A single CX is an involution and
    hides the difference; a longer sequence does not, so the transvections are
    composed in reverse here.

    The inverse map is affine over GF(2), so the table is built from the images
    of the ``n_wires`` basis vectors by doubling: entry ``i`` is the XOR of the
    images of the wires set in ``i``. That is ``n_wires`` vector operations on
    the whole table rather than one pass per basis index.
    """

    canonical_controls = tuple(int(wire) for wire in controls)
    canonical_targets = tuple(int(wire) for wire in targets)
    if len(canonical_controls) != len(canonical_targets):
        raise ValueError("a CX sequence needs one target per control")
    table_dtype = _cx_sequence_index_dtype(n_wires)
    key = (n_wires, canonical_controls, canonical_targets, str(device), table_dtype)
    cached = _CX_SEQUENCE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    images = torch.tensor(
        [1 << (n_wires - 1 - wire) for wire in range(n_wires)],
        dtype=torch.int64,
        device=device,
    )
    for control, target in zip(
        reversed(canonical_controls), reversed(canonical_targets), strict=True
    ):
        control_mask = 1 << (n_wires - 1 - control)
        target_mask = 1 << (n_wires - 1 - target)
        hit = (images & control_mask) != 0
        images = torch.where(hit, images ^ target_mask, images)
    # Doubling appends the new bit at the high end of the offset, so the first
    # step fixes the state index's least significant bit, which belongs to the
    # last wire. Consuming the wires in reverse is what keeps the bit order.
    table = torch.zeros(1, dtype=torch.int64, device=device)
    for wire in reversed(range(n_wires)):
        table = torch.cat((table, table ^ images[wire]))
    return _store_cx_sequence_index(key, table.to(table_dtype))


def _apply_cx_sequence_gather(
    state: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply a whole CX sequence as one gather over the amplitude axis."""

    index = _cx_sequence_permutation_index(
        controls, targets, n_wires, device=state.device, dtype=state.dtype
    )
    gathered = torch.index_select(state.reshape(state.shape[0], -1), 1, index)
    return gathered.reshape(state.shape)


def _apply_fixed_permutation(
    state: torch.Tensor,
    name: str,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply fixed basis permutations without constructing gate matrices."""

    tensor = state.reshape((state.shape[0],) + (2,) * n_wires)
    if name == "x":
        return torch.flip(tensor, dims=(int(wires[0]) + 1,)).reshape(state.shape)
    if name == "swap":
        first, second = (int(wire) + 1 for wire in wires)
        return tensor.transpose(first, second).reshape(state.shape)
    if name == "cx":
        return _apply_cx_permutation(state, wires, n_wires)
    raise ValueError(f"unsupported fixed permutation gate {name!r}")


def _apply_single_qubit_fixed(
    state: torch.Tensor,
    name: str,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Apply a fixed signed one-qubit permutation without batched matmul."""

    axis = int(wire) + 1
    tensor = state.reshape((state.shape[0],) + (2,) * n_wires)
    zero, one = tensor.unbind(dim=axis)
    if name == "y":
        return torch.stack((-1j * one, 1j * zero), dim=axis).reshape(state.shape)
    raise ValueError(f"unsupported fixed one-qubit gate {name!r}")


def _apply_rx_rz_loop(
    state: torch.Tensor,
    step: _StatevectorRXRZLoopStep,
    n_wires: int,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor:
    from ..triton_kernels import repeated_rx_rz

    rx_angles = []
    rz_angles = []
    for rx_instruction, rz_instruction in step.pairs:
        rx = _gate_parameter_tensor(
            rx_instruction, state, parameter_bindings=parameter_bindings
        )
        rz = _gate_parameter_tensor(
            rz_instruction, state, parameter_bindings=parameter_bindings
        )
        if rx is None or rz is None:
            raise RuntimeError("compiled RX/RZ loop is missing a gate parameter")
        rx_angles.append(rx[:, 0])
        rz_angles.append(rz[:, 0])
    rx_values = torch.stack(rx_angles, dim=1)
    rz_values = torch.stack(rz_angles, dim=1)
    rest_axes = tuple(wire + 1 for wire in range(n_wires) if wire != step.wire)
    permutation = (0,) + rest_axes + (step.wire + 1,)
    inverse = [0] * len(permutation)
    for axis, source in enumerate(permutation):
        inverse[source] = axis
    tensor = state.reshape((state.shape[0],) + (2,) * n_wires)
    paired = tensor.permute(permutation).reshape(state.shape[0], -1, 2)
    output = repeated_rx_rz(paired, rx_values, rz_values)
    if not isinstance(output, torch.Tensor):
        raise TypeError("RX/RZ loop kernel must return a tensor")
    return (
        output.reshape((state.shape[0],) + (2,) * n_wires)
        .permute(tuple(inverse))
        .reshape(state.shape[0], -1)
    )


def _gate_parameter_tensor(
    instruction: Instruction,
    state: torch.Tensor,
    *,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    slots = getattr(instruction, "parameter_slots", ())
    constants = getattr(instruction, "parameter_constants", ())
    direct_values = None
    if parameter_bindings is not None and slots:
        direct_values = tuple(
            parameter_bindings[slot] if slot >= 0 else constants[index]
            for index, slot in enumerate(slots)
        )
    return _parameter_tensor(
        instruction.name,
        instruction.params,
        bsz=state.shape[0],
        device=state.device,
        complex_dtype=state.dtype,
        direct_values=direct_values,
    )


def _fused_gate_matrix(
    step: _StatevectorFusedGateStep,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor:
    """Compose gates in execution order before scanning the statevector once."""

    matrices = tuple(
        _gate_matrix(
            instruction,
            bsz=bsz,
            device=device,
            dtype=dtype,
            parameter_bindings=parameter_bindings,
        )
        for instruction in step.instructions
    )
    combined = matrices[0]
    for matrix in matrices[1:]:
        if combined.ndim == 2 and matrix.ndim == 2:
            combined = matrix @ combined
            continue
        if combined.ndim == 2:
            combined = combined.expand(bsz, -1, -1)
        if matrix.ndim == 2:
            matrix = matrix.expand(bsz, -1, -1)
        combined = torch.bmm(matrix, combined)
    return combined


def _compose_gate_matrices(matrices: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compose matrices in circuit execution order with batch broadcasting."""

    if not matrices:
        raise ValueError("at least one gate matrix is required")
    combined = matrices[0]
    for matrix in matrices[1:]:
        combined = matrix @ combined
    return combined


def _zero_basis_local_indices(
    compressed_start: int,
    compressed_end: int,
    wires: Sequence[int],
    *,
    n_wires: int,
    rank_bits: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand compressed indices with zero bits at the selected local wires."""

    positions = sorted(n_wires - int(wire) - 1 - rank_bits for wire in wires)
    indices = torch.arange(
        compressed_start, compressed_end, dtype=torch.long, device=device
    )
    for position in positions:
        low_mask = (1 << position) - 1
        low = indices & low_mask
        indices = ((indices - low) << 1) | low
    return indices


def _basis_indices_for_wires(
    global_indices: torch.Tensor, *, n_wires: int, wires: Sequence[int]
) -> torch.Tensor:
    """Extract the selected wire bits as compact basis indices."""

    wires = tuple(int(wire) for wire in wires)
    basis = torch.zeros_like(global_indices, dtype=torch.long)
    for position, wire in enumerate(wires):
        bit = (global_indices >> (n_wires - wire - 1)) & 1
        basis |= bit.to(dtype=torch.long) << (len(wires) - position - 1)
    return basis


def _wire_mask(n_wires: int, wire: int) -> int:
    """Return the global statevector bit mask for one logical wire."""

    return 1 << (int(n_wires) - int(wire) - 1)


def _basis_offset(n_wires: int, wires: Sequence[int], basis_index: int) -> int:
    """Expand a compact gate-basis index into a global statevector offset."""

    wires = tuple(int(wire) for wire in wires)
    offset = 0
    for position, wire in enumerate(wires):
        if (int(basis_index) >> (len(wires) - position - 1)) & 1:
            offset |= _wire_mask(n_wires, wire)
    return offset


def _apply_diagonal_gate_eager(
    amplitudes: torch.Tensor,
    diagonal: torch.Tensor,
    global_indices: torch.Tensor,
    wires: Sequence[int],
    *,
    n_wires: int,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Apply diagonal entries to amplitudes selected by global basis indices."""

    basis = _basis_indices_for_wires(global_indices, n_wires=n_wires, wires=wires)
    factors = diagonal[basis].unsqueeze(0) if diagonal.ndim == 1 else diagonal[:, basis]
    out = torch.empty_like(amplitudes) if output is None else output
    out.copy_(amplitudes * factors)
    return out, factors.numel() * factors.element_size()


def _combine_rank_pair_gate_eager(
    local: torch.Tensor,
    remote: torch.Tensor,
    matrix: torch.Tensor,
    *,
    rank_basis: int,
) -> tuple[torch.Tensor, int]:
    """Combine one local and remote amplitude block for a sharded one-wire gate."""

    basis_zero, basis_one = (local, remote) if rank_basis == 0 else (remote, local)
    updated = basis_zero * matrix[rank_basis, 0] + basis_one * matrix[rank_basis, 1]
    scratch_bytes = 3 * updated.numel() * updated.element_size()
    return updated, scratch_bytes


def _combine_gate_basis_blocks_eager(
    basis_inputs: Sequence[torch.Tensor],
    matrix: torch.Tensor,
    output_basis: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Combine already-located input blocks for each gate basis state."""

    def coefficients(basis: int) -> torch.Tensor:
        if matrix.ndim == 2:
            return matrix[output_basis, basis].reshape(1, -1)
        if matrix.ndim == 3:
            return matrix[:, output_basis, basis]
        raise ValueError("gate matrix must have rank 2 or 3")

    updated = basis_inputs[0] * coefficients(0)
    for basis, input_values in enumerate(basis_inputs[1:], start=1):
        updated = updated + input_values * coefficients(basis)
    return updated, 4 * updated.numel() * updated.element_size()


def _apply_gate_basis_vectors_eager(
    values: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Apply a gate matrix to already-grouped basis vectors."""

    updated = values @ matrix.transpose(-2, -1)
    scratch_bytes = (values.numel() + updated.numel()) * values.element_size()
    return updated, scratch_bytes


def _apply_local_gate_eager(
    amplitudes: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    n_wires: int,
    rank_bits: int,
    chunk_amplitudes: int,
    output: torch.Tensor | None = None,
    validate_indices: bool = False,
) -> tuple[torch.Tensor, int]:
    """Apply a rank-local gate with PyTorch tensor operations."""

    wires = tuple(int(wire) for wire in wires)
    gate_dim = 2 ** len(wires)
    out = torch.empty_like(amplitudes) if output is None else output
    output_bytes = out.numel() * out.element_size()

    local_offsets = torch.tensor(
        [
            _basis_offset(n_wires, wires, basis) >> rank_bits
            for basis in range(gate_dim)
        ],
        dtype=torch.long,
        device=out.device,
    )
    peak = 0
    local_count = amplitudes.shape[-1]
    chunk_bases = max(1, chunk_amplitudes // gate_dim)
    basis_count = local_count >> len(wires)
    for start in range(0, basis_count, chunk_bases):
        end = min(basis_count, start + chunk_bases)
        bases = _zero_basis_local_indices(
            start,
            end,
            wires,
            n_wires=n_wires,
            rank_bits=rank_bits,
            device=out.device,
        )
        local_required = bases[:, None] | local_offsets[None, :]
        if validate_indices and (
            int(local_required.min()) < 0 or int(local_required.max()) >= local_count
        ):
            raise ValueError("local gate requires cross-shard amplitudes")
        values = amplitudes[:, local_required]
        updated, working_bytes = _apply_gate_basis_vectors_eager(values, matrix)
        out[:, local_required] = updated
        peak = max(peak, output_bytes + working_bytes)
    return out, peak


def _batched_rx_ry_rz_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Build RX->RY->RZ matrices for many regions with one tensor graph."""

    rx, ry, rz = angles.unbind(dim=-1)
    cos_x, sin_x = torch.cos(0.5 * rx), torch.sin(0.5 * rx)
    cos_y, sin_y = torch.cos(0.5 * ry), torch.sin(0.5 * ry)
    phase_neg = torch.complex(torch.cos(-0.5 * rz), torch.sin(-0.5 * rz))
    phase_pos = torch.complex(torch.cos(0.5 * rz), torch.sin(0.5 * rz))
    m00 = complex_mul(torch.complex(cos_y * cos_x, sin_y * sin_x), phase_neg)
    m01 = complex_mul(torch.complex(-sin_y * cos_x, -cos_y * sin_x), phase_neg)
    m10 = complex_mul(torch.complex(sin_y * cos_x, -cos_y * sin_x), phase_pos)
    m11 = complex_mul(torch.complex(cos_y * cos_x, -sin_y * sin_x), phase_pos)
    return torch.stack((m00, m01, m10, m11), dim=-1).reshape(*angles.shape[:-1], 2, 2)


def _batched_rotation_sequence_matrices(
    angles: torch.Tensor,
    *,
    names: tuple[str, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compose equal-topology rotation regions with one operation graph."""

    region_count, bsz, gate_count = angles.shape
    if not names or gate_count != len(names) or not set(names) <= {"rx", "ry", "rz"}:
        raise ValueError("batched rotation sequences require RX/RY/RZ topology")
    flat_matrices = []
    for index, name in enumerate(names):
        builder = GATE_MAT_DICT[name]
        if not callable(builder):
            raise TypeError(f"Rotation {name!r} requires a matrix builder.")
        flat_matrices.append(builder(angles[..., index].reshape(-1, 1)).to(dtype=dtype))
    combined = flat_matrices[0]
    for matrix in flat_matrices[1:]:
        combined = torch.bmm(matrix, combined)
    return combined.reshape(region_count, bsz, 2, 2)


def _bits_from_indices(indices: torch.Tensor, n_wires: int) -> torch.Tensor:
    shifts = torch.arange(n_wires - 1, -1, -1, device=indices.device)
    return ((indices.unsqueeze(-1) >> shifts) & 1).to(torch.int64)
