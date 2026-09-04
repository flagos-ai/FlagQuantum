"""Local statevector numerical execution.

Simulation owns the numerical loop. ``Circuit`` remains a thin public facade
and temporarily owns its lifecycle caches while the directory migration is in
progress.
"""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING

import torch

from ..circuit_statevector import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_cx_permutation,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_rx_rz_loop,
    _apply_single_qubit_fixed,
    _batched_rotation_sequence_matrices,
    _batched_rx_ry_rz_matrices,
    _canonical_name,
    _compile_statevector_program,
    _fused_gate_matrix,
    _gate_matrix,
    _gate_parameter_tensor,
    _StatevectorCXSequenceStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorRXRZLoopStep,
    _triton_parameterized_single_qubit_matrix_enabled,
    _triton_ry_rz_pair_enabled,
    _triton_single_qubit_loop_enabled,
    _triton_single_qubit_matrix_enabled,
)
from ..core.ir import CircuitIR, Instruction
from ..core.runtime_config import runtime_config

if TYPE_CHECKING:
    from ..circuit import Circuit


def _gate_parameters(
    circuit: Circuit,
    instruction: Instruction,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    cacheable = parameter_bindings is None and all(
        isinstance(value, Number) for value in instruction.params.values()
    )
    if not cacheable:
        return _gate_parameter_tensor(
            instruction,
            state,
            parameter_bindings=parameter_bindings,
        )
    key = (id(instruction), str(state.device), state.dtype, int(state.shape[0]))
    cached = circuit._statevector_constant_parameters.get(key)
    if cached is None:
        cached = _gate_parameter_tensor(instruction, state, parameter_bindings=None)
        if cached is not None:
            circuit._statevector_constant_parameters[key] = cached
    return cached


def _cx_sequence_masks(
    circuit: Circuit,
    step: _StatevectorCXSequenceStep,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (step.controls, step.targets, str(device))
    cached = circuit._statevector_cx_masks.get(key)
    if cached is None:
        controls = torch.tensor(
            [1 << (circuit.n_wires - 1 - wire) for wire in step.controls],
            dtype=torch.int64,
            device=device,
        )
        targets = torch.tensor(
            [1 << (circuit.n_wires - 1 - wire) for wire in step.targets],
            dtype=torch.int64,
            device=device,
        )
        cached = (controls, targets, controls.flip(0), targets.flip(0))
        circuit._statevector_cx_masks[key] = cached
    return cached


def _fused_constant_matrix(
    circuit: Circuit,
    step: _StatevectorFusedGateStep,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    cacheable = (
        len(step.wires) == 1
        and parameter_bindings is None
        and all(
            instruction.matrix is None
            and all(isinstance(value, Number) for value in instruction.params.values())
            for instruction in step.instructions
        )
    )
    if not cacheable:
        return None
    key = (id(step), str(state.device), state.dtype, int(state.shape[0]))
    cached = circuit._statevector_fused_matrices.get(key)
    if cached is None:
        cached = _fused_gate_matrix(
            step,
            bsz=state.shape[0],
            device=state.device,
            dtype=state.dtype,
            parameter_bindings=None,
        ).detach()
        circuit._statevector_fused_matrices[key] = cached
    return cached


def state(circuit: Circuit, *, refresh: bool = False) -> torch.Tensor:
    """Execute one Circuit through the Simulation-owned statevector loop."""

    if circuit._state_cache is not None and not refresh:
        return circuit._state_cache
    with runtime_config(circuit.runtime_config):
        output = circuit.initial_state()
        binding_source = getattr(circuit, "_parameter_bindings", None)
        parameter_bindings = None if binding_source is None else binding_source.values()
        enable_triton_loop = (
            _triton_single_qubit_loop_enabled()
            and output.is_cuda
            and output.dtype == torch.complex64
        )
        program_key = ("statevector", "triton_rx_rz_loop", enable_triton_loop)
        program = circuit._backend_programs.get(program_key)
        if program is None:
            program = _compile_statevector_program(
                circuit._instructions,
                circuit.n_wires,
                enable_triton_loop=enable_triton_loop,
            )
            circuit._backend_programs[program_key] = program
        fused_steps = tuple(
            step for step in program if isinstance(step, _StatevectorRXRZLoopStep)
        )
        triton_ry_rz_pairs = sum(
            isinstance(step, _StatevectorFusedGateStep)
            and tuple(item.name for item in step.instructions) == ("ry", "rz")
            for step in program
        )
        circuit._last_statevector_runtime = {
            "triton_single_qubit_loop_enabled": enable_triton_loop,
            "triton_single_qubit_loop_regions": len(fused_steps),
            "triton_single_qubit_loop_gates": sum(
                2 * len(step.pairs) for step in fused_steps
            ),
            "triton_ry_rz_pair_candidates": triton_ry_rz_pairs,
            "triton_ry_rz_pair_executed": 0,
            "triton_single_qubit_matrix_regions": 0,
            "diagonal_elementwise_gates": sum(
                isinstance(step, _StatevectorGateStep)
                and _canonical_name(step.instruction.name)
                in _DIAGONAL_STATEVECTOR_GATES
                for step in program
            ),
            "permutation_gates": sum(
                isinstance(step, _StatevectorGateStep)
                and _canonical_name(step.instruction.name) in {"x", "cx", "swap"}
                for step in program
            )
            + sum(
                len(step.controls)
                for step in program
                if isinstance(step, _StatevectorCXSequenceStep)
            ),
            "triton_cx_sequence_regions": sum(
                isinstance(step, _StatevectorCXSequenceStep) for step in program
            ),
            "fixed_single_qubit_specialized_gates": sum(
                isinstance(step, _StatevectorGateStep)
                and _canonical_name(step.instruction.name) == "y"
                for step in program
            ),
            "fused_gate_regions": sum(
                isinstance(step, _StatevectorFusedGateStep) for step in program
            ),
            "fused_gate_count": sum(
                len(step.instructions)
                for step in program
                if isinstance(step, _StatevectorFusedGateStep)
            ),
            "dependency_reordered_single_qubit_regions": sum(
                isinstance(step, _StatevectorFusedGateStep)
                and step.dependency_reordered
                for step in program
            ),
            "statevector_apply_count": len(program),
        }
        rx_ry_rz_steps = tuple(
            step
            for step in program
            if isinstance(step, _StatevectorFusedGateStep)
            and tuple(item.name for item in step.instructions) == ("rx", "ry", "rz")
            and len(step.wires) == 1
        )
        batched_rx_ry_rz_matrices: dict[int, torch.Tensor] = {}
        if rx_ry_rz_steps:
            region_angles = []
            for step in rx_ry_rz_steps:
                parameters = tuple(
                    _gate_parameters(
                        circuit,
                        instruction,
                        output,
                        parameter_bindings,
                    )
                    for instruction in step.instructions
                )
                if any(parameter is None for parameter in parameters):
                    region_angles = []
                    break
                region_angles.append(
                    torch.stack(
                        tuple(parameter[:, 0] for parameter in parameters), dim=-1
                    )
                )
            if region_angles:
                matrices = _batched_rx_ry_rz_matrices(
                    torch.stack(region_angles, dim=0)
                ).to(dtype=output.dtype)
                batched_rx_ry_rz_matrices = {
                    id(step): matrices[index]
                    for index, step in enumerate(rx_ry_rz_steps)
                }
        circuit._last_statevector_runtime["batched_rx_ry_rz_regions"] = len(
            batched_rx_ry_rz_matrices
        )
        batched_rotation_matrices: dict[int, torch.Tensor] = {}
        rotation_patterns = {
            tuple(item.name for item in step.instructions)
            for step in program
            if isinstance(step, _StatevectorFusedGateStep)
            and len(step.instructions) >= 2
            and all(item.name in {"rx", "ry", "rz"} for item in step.instructions)
        }
        for names in rotation_patterns:
            rotation_steps = tuple(
                step
                for step in program
                if isinstance(step, _StatevectorFusedGateStep)
                and tuple(item.name for item in step.instructions) == names
            )
            region_angles = []
            for step in rotation_steps:
                parameters = tuple(
                    _gate_parameters(
                        circuit,
                        instruction,
                        output,
                        parameter_bindings,
                    )
                    for instruction in step.instructions
                )
                if any(parameter is None for parameter in parameters):
                    region_angles = []
                    break
                region_angles.append(
                    torch.stack(
                        tuple(parameter[:, 0] for parameter in parameters), dim=-1
                    )
                )
            if region_angles:
                matrices = _batched_rotation_sequence_matrices(
                    torch.stack(region_angles, dim=0),
                    names=names,
                    dtype=output.dtype,
                )
                batched_rotation_matrices.update(
                    {
                        id(step): matrices[index]
                        for index, step in enumerate(rotation_steps)
                    }
                )
        circuit._last_statevector_runtime["batched_rotation_sequence_regions"] = len(
            batched_rotation_matrices
        )
        for step in program:
            if isinstance(step, _StatevectorCXSequenceStep):
                if output.is_cuda and output.dtype == torch.complex64:
                    from .triton_kernels import cx_sequence

                    (
                        control_masks,
                        target_masks,
                        reverse_control_masks,
                        reverse_target_masks,
                    ) = _cx_sequence_masks(circuit, step, output.device)
                    output = cx_sequence(
                        output,
                        control_masks=control_masks,
                        target_masks=target_masks,
                        reverse_control_masks=reverse_control_masks,
                        reverse_target_masks=reverse_target_masks,
                        n_wires=circuit.n_wires,
                    )
                else:
                    for control, target in zip(
                        step.controls, step.targets, strict=True
                    ):
                        output = _apply_cx_permutation(
                            output, (control, target), circuit.n_wires
                        )
                continue
            if isinstance(step, _StatevectorRXRZLoopStep):
                output = _apply_rx_rz_loop(
                    output,
                    step,
                    circuit.n_wires,
                    parameter_bindings,
                )
                continue
            if isinstance(step, _StatevectorFusedGateStep):
                if (
                    output.is_cuda
                    and output.dtype == torch.complex64
                    and _triton_ry_rz_pair_enabled()
                    and tuple(item.name for item in step.instructions) == ("ry", "rz")
                ):
                    ry_angles = _gate_parameters(
                        circuit,
                        step.instructions[0],
                        output,
                        parameter_bindings,
                    )
                    rz_angles = _gate_parameters(
                        circuit,
                        step.instructions[1],
                        output,
                        parameter_bindings,
                    )
                    if (
                        ry_angles is not None
                        and rz_angles is not None
                        and not output.requires_grad
                        and not ry_angles.requires_grad
                        and not rz_angles.requires_grad
                    ):
                        from .triton_kernels import ry_rz_pair

                        output = ry_rz_pair(
                            output,
                            ry_angles,
                            rz_angles,
                            wire=step.wires[0],
                            n_wires=circuit.n_wires,
                        )
                        circuit._last_statevector_runtime[
                            "triton_ry_rz_pair_executed"
                        ] += 1
                        continue
                constant_matrix = _fused_constant_matrix(
                    circuit, step, output, parameter_bindings
                )
                if (
                    constant_matrix is not None
                    and output.is_cuda
                    and output.dtype == torch.complex64
                    and _triton_single_qubit_matrix_enabled()
                ):
                    from .triton_kernels import single_qubit_matrix

                    output = single_qubit_matrix(
                        output,
                        constant_matrix,
                        wire=step.wires[0],
                        n_wires=circuit.n_wires,
                    )
                    circuit._last_statevector_runtime[
                        "triton_single_qubit_matrix_regions"
                    ] += 1
                    continue
                matrix = batched_rotation_matrices.get(id(step))
                if matrix is None:
                    matrix = batched_rx_ry_rz_matrices.get(id(step))
                if matrix is None:
                    matrix = _fused_gate_matrix(
                        step,
                        bsz=output.shape[0],
                        device=output.device,
                        dtype=output.dtype,
                        parameter_bindings=parameter_bindings,
                    )
                if (
                    output.is_cuda
                    and output.dtype == torch.complex64
                    and len(step.wires) == 1
                    and _triton_single_qubit_matrix_enabled()
                    and (
                        not matrix.requires_grad
                        or _triton_parameterized_single_qubit_matrix_enabled()
                    )
                ):
                    from .triton_kernels import single_qubit_matrix

                    output = single_qubit_matrix(
                        output,
                        matrix,
                        wire=step.wires[0],
                        n_wires=circuit.n_wires,
                    )
                    circuit._last_statevector_runtime[
                        "triton_single_qubit_matrix_regions"
                    ] += 1
                    continue
                output = _apply_matrix(
                    output,
                    matrix,
                    step.wires,
                    circuit.n_wires,
                    layout=step.layout,
                )
                continue
            instruction = step.instruction
            name = _canonical_name(instruction.name)
            if name in {"x", "cx", "swap"}:
                output = _apply_fixed_permutation(
                    output, name, instruction.wires, circuit.n_wires
                )
            elif name == "y":
                output = _apply_single_qubit_fixed(
                    output, name, instruction.wires[0], circuit.n_wires
                )
            else:
                matrix = _gate_matrix(
                    instruction,
                    bsz=output.shape[0],
                    device=output.device,
                    dtype=output.dtype,
                    parameter_bindings=parameter_bindings,
                )
                apply_gate = (
                    _apply_diagonal_matrix
                    if name in _DIAGONAL_STATEVECTOR_GATES
                    else _apply_matrix
                )
                output = apply_gate(
                    output,
                    matrix,
                    instruction.wires,
                    circuit.n_wires,
                    layout=step.layout,
                )
    circuit._state_cache = output
    return output


def run_local_statevector(
    program: CircuitIR,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Execute validated IR on one resolved device without Runtime policy."""

    if not isinstance(program, CircuitIR):
        raise TypeError("program must be a CircuitIR")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from ..circuit import Circuit

    return Circuit.from_ir(
        program,
        bsz=batch_size,
        device=device,
        dtype=dtype,
    ).state()


__all__ = ("run_local_statevector", "state")
