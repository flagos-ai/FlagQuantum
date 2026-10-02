"""Native CPU execution for static disjoint one-qubit layers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

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
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorPreCXStep,
)

_MAX_NATIVE_FIXED_LAYER_WIRES = 11
_MAX_NATIVE_PARAMETERIZED_LAYER_WIRES = 11
_MAX_NATIVE_FUSED_ROTATION_LAYER_WIRES = 6
_NATIVE_FIXED_GATES = frozenset({"h", "s", "sdg", "x", "y", "z"})
_NATIVE_PARAMETERIZED_GATES = frozenset({"rx", "ry", "rz"})
_NATIVE_CLIFFORD_CODES = {"h": 1, "s": 2, "sdg": 3, "x": 4, "y": 5, "z": 6}


def _matrix_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    """The tensors an instruction carries in its matrix field.

    A gate holds one unitary tensor there. A channel holds the tuple of Kraus
    operators it was materialized from. Reading the field as a value keeps this
    guard from assuming which of the two it is, which is what raised
    ``AttributeError: 'tuple' object has no attribute 'requires_grad'``.
    """

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, torch.Tensor))
    return ()


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
            isinstance(value, torch.Tensor) and value.requires_grad
            # A channel instruction carries its Kraus operators in the same field,
            # so the field is read as a value rather than assumed to be one tensor.
            for instruction in instructions
            for value in _matrix_tensors(instruction.matrix)
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


def native_fused_rotation_layer_compile_enabled() -> bool:
    """Whether fused same-wire rotations may join native disjoint layers."""

    return os.getenv("FQ_CPU_NATIVE_FUSED_ROTATION_LAYER", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


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
    *,
    include_terminal_fused_regions: bool = False,
) -> list[_StatevectorPreCXStep]:
    """Compile disjoint named rotations for batch-specific native matrices."""

    def fused_rotation_instructions(
        step: _StatevectorPreCXStep,
    ) -> tuple[Instruction, ...]:
        if not (isinstance(step, _StatevectorFusedGateStep) and len(step.wires) == 1):
            return ()
        instructions = step.instructions
        if any(
            instruction.matrix is not None
            or not instruction.params
            or len(instruction.wires) != 1
            or canonical_opcode(instruction.name) not in _NATIVE_PARAMETERIZED_GATES
            for instruction in instructions
        ):
            return ()
        return instructions

    terminal_fused_start = len(program)
    if include_terminal_fused_regions:
        for candidate_index in range(len(program) - 1, -1, -1):
            if not fused_rotation_instructions(program[candidate_index]):
                break
            terminal_fused_start = candidate_index

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        group: list[_StatevectorGateStep | _StatevectorFusedGateStep] = []
        occupied: set[int] = set()
        cursor = index
        max_wires = (
            _MAX_NATIVE_FUSED_ROTATION_LAYER_WIRES
            if include_terminal_fused_regions
            and index >= terminal_fused_start
            and isinstance(program[index], _StatevectorFusedGateStep)
            else _MAX_NATIVE_PARAMETERIZED_LAYER_WIRES
        )
        while cursor < len(program) and len(group) < max_wires:
            candidate = program[cursor]
            instructions: tuple[Instruction, ...]
            if isinstance(candidate, _StatevectorGateStep):
                instructions = (candidate.instruction,)
            elif (
                isinstance(candidate, _StatevectorFusedGateStep)
                and include_terminal_fused_regions
                and cursor >= terminal_fused_start
            ):
                instructions = fused_rotation_instructions(candidate)
            else:
                break
            if not instructions or any(
                instruction.matrix is not None
                or not instruction.params
                or len(instruction.wires) != 1
                or canonical_opcode(instruction.name) not in _NATIVE_PARAMETERIZED_GATES
                for instruction in instructions
            ):
                break
            wire = int(instructions[0].wires[0])
            if any(int(instruction.wires[0]) != wire for instruction in instructions):
                break
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
    matrix_builder: (
        Callable[
            [_StatevectorGateStep | _StatevectorFusedGateStep, torch.Tensor],
            torch.Tensor,
        ]
        | None
    ) = None,
) -> torch.Tensor | None:
    """Apply one shared or batch-specific inference layer natively."""

    if (
        not (step.native_preferred or step.native_parameterized)
        or state.device.type != "cpu"
        or state.requires_grad
        or not native_cpu_one_qubit_layer_available()
    ):
        return None
    regions = tuple(
        region
        for region in step.regions
        if isinstance(region, (_StatevectorGateStep, _StatevectorFusedGateStep))
    )
    if len(regions) != len(step.regions) or not all(
        (
            len(region.instruction.wires)
            if isinstance(region, _StatevectorGateStep)
            else len(region.wires)
        )
        == 1
        for region in regions
    ):
        return None
    wires = tuple(
        int(
            region.instruction.wires[0]
            if isinstance(region, _StatevectorGateStep)
            else region.wires[0]
        )
        for region in regions
    )
    output = state if owns_state else state.clone()
    if step.native_clifford:
        if not all(isinstance(region, _StatevectorGateStep) for region in regions):
            return None
        gate_codes = tuple(
            _NATIVE_CLIFFORD_CODES[canonical_opcode(region.instruction.name)]
            for region in regions
            if isinstance(region, _StatevectorGateStep)
        )
        if fused_static_clifford_layer_(
            output,
            torch.tensor(gate_codes, dtype=torch.int8, device=state.device),
            torch.tensor(wires, dtype=torch.int64, device=state.device),
            n_wires=n_wires,
        ):
            return output
    if all(isinstance(region, _StatevectorGateStep) for region in regions):
        matrices = tuple(
            gate_matrix(
                region.instruction,
                bsz=state.shape[0],
                device=state.device,
                dtype=state.dtype,
                parameter_bindings=parameter_bindings,
            )
            for region in regions
            if isinstance(region, _StatevectorGateStep)
        )
    elif matrix_builder is not None:
        matrices = tuple(matrix_builder(region, state) for region in regions)
    else:
        return None
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
