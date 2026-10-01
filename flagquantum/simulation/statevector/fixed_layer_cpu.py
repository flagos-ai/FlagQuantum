"""Native CPU execution for static disjoint one-qubit layers."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ..gate_matrix import gate_matrix
from ..native_cpu.rotation import (
    fused_rotation_block_forward_,
    fused_static_clifford_layer_,
    native_cpu_one_qubit_layer_available,
)
from .program import (
    _StatevectorDisjointDenseStep,
    _StatevectorGateStep,
    _StatevectorPreCXStep,
)

_MAX_NATIVE_FIXED_LAYER_WIRES = 11
_MAX_NATIVE_PARAMETERIZED_LAYER_WIRES = 11
_NATIVE_FIXED_GATES = frozenset({"h", "s", "sdg", "x", "y", "z"})
_NATIVE_PARAMETERIZED_GATES = frozenset({"rx", "ry", "rz"})
_NATIVE_CLIFFORD_CODES = {"h": 1, "s": 2, "sdg": 3, "x": 4, "y": 5, "z": 6}


def _native_layer_compile_safe(
    instructions: Sequence[Instruction],
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    state: torch.Tensor,
    *,
    batch_size: int,
) -> bool:
    """Whether an inference-only native layer can preserve tensor ownership."""

    requires_grad = bool(
        state.requires_grad
        or (
            parameter_bindings is not None
            and any(value.requires_grad for value in parameter_bindings)
        )
        or any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for instruction in instructions
            for value in instruction.params.values()
        )
        or any(
            instruction.matrix is not None and instruction.matrix.requires_grad
            for instruction in instructions
        )
    )
    return bool(
        state.device.type == "cpu"
        and batch_size >= 2
        and native_cpu_one_qubit_layer_available()
        and not requires_grad
    )


def native_fixed_layer_compile_enabled(
    instructions: Sequence[Instruction],
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    state: torch.Tensor,
    *,
    batch_size: int,
) -> bool:
    """Whether static Clifford layers may use the native CPU kernel."""

    return bool(
        _native_layer_compile_safe(
            instructions, parameter_bindings, state, batch_size=batch_size
        )
        and os.getenv("FQ_CPU_NATIVE_FIXED_ONE_QUBIT_LAYER", "1").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def native_parameterized_layer_compile_enabled(
    instructions: Sequence[Instruction],
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    state: torch.Tensor,
    *,
    batch_size: int,
) -> bool:
    """Whether parameterized rotation layers may use the native CPU kernel."""

    return bool(
        _native_layer_compile_safe(
            instructions, parameter_bindings, state, batch_size=batch_size
        )
        and os.getenv("FQ_CPU_NATIVE_PARAMETERIZED_ONE_QUBIT_LAYER", "1")
        .strip()
        .lower()
        not in {"0", "false", "off", "no"}
    )


def fuse_native_fixed_one_qubit_layers(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Compile disjoint static Clifford gates for the native CPU layer kernel."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        group: list[_StatevectorGateStep] = []
        occupied: set[int] = set()
        cursor = index
        hadamards = 0
        while cursor < len(program):
            candidate = program[cursor]
            if not isinstance(candidate, _StatevectorGateStep):
                break
            instruction = candidate.instruction
            if (
                instruction.matrix is not None
                or instruction.params
                or len(instruction.wires) != 1
                or canonical_opcode(instruction.name) not in _NATIVE_FIXED_GATES
            ):
                break
            opcode = canonical_opcode(instruction.name)
            if opcode == "h" and hadamards == _MAX_NATIVE_FIXED_LAYER_WIRES:
                break
            wire = int(instruction.wires[0])
            if wire in occupied:
                break
            group.append(candidate)
            occupied.add(wire)
            hadamards += opcode == "h"
            cursor += 1
        if len(group) >= 2:
            optimized.append(
                _StatevectorDisjointDenseStep(
                    tuple(group), native_preferred=True, native_clifford=True
                )
            )
            index = cursor
            continue
        optimized.append(program[index])
        index += 1
    return optimized


def fuse_native_parameterized_one_qubit_layers(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Compile disjoint named rotations for batch-specific native matrices."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        group: list[_StatevectorGateStep] = []
        occupied: set[int] = set()
        cursor = index
        while (
            cursor < len(program) and len(group) < _MAX_NATIVE_PARAMETERIZED_LAYER_WIRES
        ):
            candidate = program[cursor]
            if not isinstance(candidate, _StatevectorGateStep):
                break
            instruction = candidate.instruction
            if (
                instruction.matrix is not None
                or not instruction.params
                or len(instruction.wires) != 1
                or canonical_opcode(instruction.name) not in _NATIVE_PARAMETERIZED_GATES
            ):
                break
            wire = int(instruction.wires[0])
            if wire in occupied:
                break
            group.append(candidate)
            occupied.add(wire)
            cursor += 1
        if len(group) >= 2:
            optimized.append(
                _StatevectorDisjointDenseStep(tuple(group), native_parameterized=True)
            )
            index = cursor
            continue
        optimized.append(program[index])
        index += 1
    return optimized


def apply_native_fixed_one_qubit_layer(
    step: _StatevectorDisjointDenseStep,
    state: torch.Tensor,
    *,
    n_wires: int,
    owns_state: bool,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    """Apply one shared or batch-specific inference layer natively."""

    if (
        not (step.native_preferred or step.native_parameterized)
        or state.device.type != "cpu"
        or state.requires_grad
        or not native_cpu_one_qubit_layer_available()
        or not all(isinstance(region, _StatevectorGateStep) for region in step.regions)
    ):
        return None
    gate_steps = tuple(
        region for region in step.regions if isinstance(region, _StatevectorGateStep)
    )
    wires = tuple(int(region.instruction.wires[0]) for region in gate_steps)
    output = state if owns_state else state.clone()
    if step.native_clifford:
        gate_codes = tuple(
            _NATIVE_CLIFFORD_CODES[canonical_opcode(region.instruction.name)]
            for region in gate_steps
        )
        if fused_static_clifford_layer_(
            output,
            torch.tensor(gate_codes, dtype=torch.int8, device=state.device),
            torch.tensor(wires, dtype=torch.int64, device=state.device),
            n_wires=n_wires,
        ):
            return output
    matrices = tuple(
        gate_matrix(
            region.instruction,
            bsz=state.shape[0],
            device=state.device,
            dtype=state.dtype,
            parameter_bindings=parameter_bindings,
        )
        for region in gate_steps
    )
    if (
        len(matrices) != len(wires)
        or len(wires) < 2
        or any(matrix.ndim not in {2, 3} or matrix.requires_grad for matrix in matrices)
    ):
        return None
    uses_batch_matrices = any(matrix.ndim == 3 for matrix in matrices)
    if uses_batch_matrices:
        if any(
            matrix.ndim == 3 and matrix.shape[0] != state.shape[0]
            for matrix in matrices
        ):
            return None
        matrix_tensor = torch.stack(
            tuple(
                matrix.expand(state.shape[0], -1, -1) if matrix.ndim == 2 else matrix
                for matrix in matrices
            ),
            dim=1,
        ).contiguous()
    else:
        matrix_tensor = torch.stack(matrices).contiguous()
    start = 0
    while start < len(wires):
        width = min(_MAX_NATIVE_FIXED_LAYER_WIRES, len(wires) - start)
        if len(wires) - start - width == 1:
            width -= 1
        stop = start + width
        if not fused_rotation_block_forward_(
            output,
            matrix_tensor.narrow(-3, start, width),
            torch.tensor(wires[start:stop], dtype=torch.int64, device=state.device),
            n_wires=n_wires,
        ):
            return None
        start = stop
    return output
