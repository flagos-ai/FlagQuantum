"""Private statevector compilation and tensor helpers for :mod:`circuit`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from .core.ir import Instruction
from .core.operator_schema import canonical_opcode, get_operator_schema
from .core.parameters import value_to_tensor
from .ops.matrices import GATE_MAT_DICT, get_global_precision

_FIXED_GATE_CACHE: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
_STATEVECTOR_LAYOUT_CACHE: dict[
    tuple[int, tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]
] = {}
_DIAGONAL_STATEVECTOR_GATES = frozenset(
    {"z", "s", "sdg", "t", "tdg", "rz", "p", "phase", "u1", "cz", "cphase", "rzz"}
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


@dataclass(frozen=True)
class _StatevectorCXSequenceStep:
    controls: tuple[int, ...]
    targets: tuple[int, ...]


def _triton_single_qubit_loop_enabled() -> bool:
    return os.getenv("FQ_TRITON_SINGLE_QUBIT_LOOP", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _triton_ry_rz_pair_enabled() -> bool:
    return os.getenv("FQ_TRITON_RY_RZ_PAIR", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _triton_single_qubit_matrix_enabled() -> bool:
    return os.getenv("FQ_TRITON_SINGLE_QUBIT_MATRIX", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _triton_parameterized_single_qubit_matrix_enabled() -> bool:
    return os.getenv(
        "FQ_TRITON_PARAMETERIZED_SINGLE_QUBIT_MATRIX", "0"
    ).strip().lower() in {"1", "true", "on", "yes"}


def _compile_statevector_program(
    instructions: Sequence[Instruction], n_wires: int, *, enable_triton_loop: bool
) -> tuple[
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorFusedGateStep
    | _StatevectorCXSequenceStep,
    ...,
]:
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
            wire_order: list[int] = []
            for instruction in segment:
                wire = instruction.wires[0]
                if wire not in by_wire:
                    by_wire[wire] = []
                    wire_order.append(wire)
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
                for wire in wire_order:
                    items = by_wire[wire]
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
            by_wire: dict[int, list[_StatevectorGateStep]] = {}
            wire_order: list[int] = []
            while cursor < len(program):
                candidate = program[cursor]
                if not (
                    isinstance(candidate, _StatevectorGateStep)
                    and len(candidate.instruction.wires) == 1
                ):
                    break
                wire = candidate.instruction.wires[0]
                if wire not in by_wire:
                    by_wire[wire] = []
                    wire_order.append(wire)
                by_wire[wire].append(candidate)
                cursor += 1
            for wire in wire_order:
                wire_steps = by_wire[wire]
                if len(wire_steps) == 1:
                    fused_program.append(wire_steps[0])
                else:
                    wire_step_ids = {id(item) for item in wire_steps}
                    positions = [
                        position
                        for position, candidate in enumerate(program[index:cursor])
                        if id(candidate) in wire_step_ids
                    ]
                    fused_program.append(
                        _StatevectorFusedGateStep(
                            instructions=tuple(item.instruction for item in wire_steps),
                            wires=(wire,),
                            layout=wire_steps[0].layout,
                            dependency_reordered=any(
                                right != left + 1
                                for left, right in zip(positions, positions[1:])
                            ),
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
            fused_program.append(
                _StatevectorFusedGateStep(
                    instructions=tuple(item.instruction for item in group),
                    wires=tuple(step.instruction.wires),
                    layout=step.layout,
                )
            )
        index = cursor
    optimized_program: list[
        _StatevectorGateStep
        | _StatevectorRXRZLoopStep
        | _StatevectorFusedGateStep
        | _StatevectorCXSequenceStep
    ] = []
    index = 0
    while index < len(fused_program):
        step = fused_program[index]
        if not (
            isinstance(step, _StatevectorGateStep)
            and _canonical_name(step.instruction.name) == "cx"
        ):
            optimized_program.append(step)
            index += 1
            continue
        cx_steps = [step]
        cursor = index + 1
        while cursor < len(fused_program):
            candidate = fused_program[cursor]
            if not (
                isinstance(candidate, _StatevectorGateStep)
                and _canonical_name(candidate.instruction.name) == "cx"
            ):
                break
            cx_steps.append(candidate)
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
    return tuple(optimized_program)


def _canonical_name(name: str) -> str:
    return canonical_opcode(name)


def _normalize_wires(wires: Iterable[int] | int) -> tuple[int, ...]:
    if isinstance(wires, int):
        return (wires,)
    return tuple(int(wire) for wire in wires)


def _parameter_tensor(
    name: str,
    params: Mapping[str, Any],
    *,
    bsz: int,
    device: torch.device | str,
    complex_dtype: torch.dtype | None = None,
    direct_values: tuple[Any, ...] | None = None,
) -> torch.Tensor | None:
    schema = get_operator_schema(name)
    names = schema.parameters if schema is not None else ()
    if not names:
        return None

    if direct_values is None:
        values = []
        for param_name in names:
            if param_name not in params:
                return None
            values.append(params[param_name])
    else:
        values = list(direct_values)

    tensors = [value_to_tensor(value, device=device) for value in values]
    stacked = torch.stack(
        [t.reshape(()) if t.ndim == 0 else t for t in tensors], dim=-1
    )
    complex_dtype = complex_dtype or get_global_precision()
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    stacked = stacked.to(device=device, dtype=real_dtype)
    if stacked.ndim == 1:
        stacked = stacked.reshape(1, -1)
    if stacked.shape[0] == 1 and bsz > 1:
        stacked = stacked.expand(bsz, -1)
    return stacked


def _gate_matrix(
    instruction: Instruction,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    parameter_bindings: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    dtype = dtype or get_global_precision()
    if instruction.matrix is not None:
        matrix = instruction.matrix
        matrix = getattr(matrix, "tensor", matrix)
        return torch.as_tensor(matrix, dtype=dtype, device=device).reshape(
            2 ** len(instruction.wires), 2 ** len(instruction.wires)
        )

    name = _canonical_name(instruction.name)
    gate = GATE_MAT_DICT[name]
    slots = getattr(instruction, "parameter_slots", ())
    constants = getattr(instruction, "parameter_constants", ())
    direct_values = None
    if parameter_bindings is not None and slots:
        direct_values = tuple(
            parameter_bindings[slot] if slot >= 0 else constants[index]
            for index, slot in enumerate(slots)
        )
    params = _parameter_tensor(
        name,
        instruction.params,
        bsz=bsz,
        device=device,
        complex_dtype=dtype,
        direct_values=direct_values,
    )
    if callable(gate):
        if params is None:
            raise ValueError(f"Gate {name!r} requires parameters.")
        return gate(params).to(device=device, dtype=dtype)
    key = (name, str(torch.device(device)), dtype)
    cached = _FIXED_GATE_CACHE.get(key)
    if cached is None:
        cached = gate.to(device=device, dtype=dtype)
        _FIXED_GATE_CACHE[key] = cached
    return cached


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
    bsz = state.shape[0]
    k = len(wires)
    dim = 2**k
    if layout is None:
        layout = _statevector_layout(n_wires, wires)
    perm, inv_perm = layout

    tensor = state.reshape((bsz,) + (2,) * n_wires).permute(perm)
    flat = tensor.reshape(bsz, dim, -1)
    if matrix.ndim == 2:
        matrix = matrix.expand(bsz, -1, -1)
    out = torch.bmm(matrix, flat)
    return out.reshape((bsz,) + (2,) * n_wires).permute(inv_perm).reshape(bsz, -1)


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
    if layout is None:
        layout = _statevector_layout(n_wires, wires)
    perm, inv_perm = layout
    tensor = state.reshape((bsz,) + (2,) * n_wires).permute(perm)
    flat = tensor.reshape(bsz, dim, -1)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    if diagonal.ndim == 1:
        diagonal = diagonal.expand(bsz, -1)
    out = flat * diagonal.unsqueeze(-1)
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
    from .simulation.triton_kernels import repeated_rx_rz

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


def _batched_rx_ry_rz_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Build RX->RY->RZ matrices for many regions with one tensor graph."""

    rx, ry, rz = angles.unbind(dim=-1)
    cos_x, sin_x = torch.cos(0.5 * rx), torch.sin(0.5 * rx)
    cos_y, sin_y = torch.cos(0.5 * ry), torch.sin(0.5 * ry)
    phase_neg = torch.exp(-0.5j * rz)
    phase_pos = phase_neg.conj()
    m00 = torch.complex(cos_y * cos_x, sin_y * sin_x) * phase_neg
    m01 = torch.complex(-sin_y * cos_x, -cos_y * sin_x) * phase_neg
    m10 = torch.complex(sin_y * cos_x, -cos_y * sin_x) * phase_pos
    m11 = torch.complex(cos_y * cos_x, -sin_y * sin_x) * phase_pos
    return torch.stack((m00, m01, m10, m11), dim=-1).reshape(*angles.shape[:-1], 2, 2)


def _bits_from_indices(indices: torch.Tensor, n_wires: int) -> torch.Tensor:
    shifts = torch.arange(n_wires - 1, -1, -1, device=indices.device)
    return ((indices.unsqueeze(-1) >> shifts) & 1).to(torch.int64)
