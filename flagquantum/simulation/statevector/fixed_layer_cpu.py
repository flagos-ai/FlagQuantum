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
    native_cpu_one_qubit_layer_available,
)
from .program import (
    _StatevectorDisjointDenseStep,
    _StatevectorGateStep,
    _StatevectorPreCXStep,
)

_MAX_NATIVE_FIXED_LAYER_WIRES = 11
_NATIVE_FIXED_GATES = frozenset({"h", "s", "sdg", "x", "y", "z"})


def native_fixed_layer_compile_enabled(
    instructions: Sequence[Instruction],
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    state: torch.Tensor,
    *,
    batch_size: int,
) -> bool:
    """Whether inference-only native layers are safe and available."""

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
        and os.getenv("FQ_CPU_NATIVE_FIXED_ONE_QUBIT_LAYER", "1").strip().lower()
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
        while cursor < len(program) and len(group) < _MAX_NATIVE_FIXED_LAYER_WIRES:
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
            wire = int(instruction.wires[0])
            if wire in occupied:
                break
            group.append(candidate)
            occupied.add(wire)
            cursor += 1
        if len(group) >= 2:
            optimized.append(
                _StatevectorDisjointDenseStep(tuple(group), native_preferred=True)
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
) -> torch.Tensor | None:
    """Apply one shared, inference-only CPU layer through the native operator."""

    if (
        not step.native_preferred
        or state.device.type != "cpu"
        or state.requires_grad
        or not native_cpu_one_qubit_layer_available()
        or not all(isinstance(region, _StatevectorGateStep) for region in step.regions)
    ):
        return None
    gate_steps = tuple(
        region for region in step.regions if isinstance(region, _StatevectorGateStep)
    )
    matrices = tuple(
        gate_matrix(
            region.instruction,
            bsz=state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )
        for region in gate_steps
    )
    wires = tuple(int(region.instruction.wires[0]) for region in gate_steps)
    if (
        len(matrices) != len(wires)
        or not 2 <= len(wires) <= _MAX_NATIVE_FIXED_LAYER_WIRES
        or any(matrix.ndim != 2 or matrix.requires_grad for matrix in matrices)
    ):
        return None
    output = state if owns_state else state.clone()
    if not fused_rotation_block_forward_(
        output,
        torch.stack(matrices).contiguous(),
        torch.tensor(wires, dtype=torch.int64, device=state.device),
        n_wires=n_wires,
    ):
        return None
    return output
