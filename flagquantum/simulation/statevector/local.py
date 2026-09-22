"""Local statevector numerical execution.

Simulation owns the numerical loop and cache contents. ``Circuit`` remains the
thin public facade and owns cache containers and mutation-time invalidation.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any

import torch

from ...core.ir import CircuitIR, Instruction
from ...core.operator_schema import canonical_opcode
from ...core.runtime_config import runtime_config
from ..gate_matrix import gate_matrix as _gate_matrix
from ..matrices import X_MATRIX, Y_MATRIX, Z_MATRIX
from ..numerics.complex_arithmetic import complex_conj, complex_mul
from .operations import (
    _CX_SEQUENCE_GATHER_MINIMUM_LENGTH,
    _apply_cx_permutation,
    _apply_cx_sequence_gather,
    _apply_diagonal_matrix,
    _apply_disjoint_diagonal_regions_cpu,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_rx_rz_loop,
    _apply_single_qubit_fixed,
    _apply_single_wire_matrix,
    _batched_rotation_sequence_matrices,
    _batched_rx_ry_rz_matrices,
    _bits_from_indices,
    _compile_statevector_program,
    _cpu_cross_wire_diagonal_fusion_enabled,
    _cpu_cx_sequence_gather_enabled,
    _cpu_disjoint_single_wire_fusion_enabled,
    _cpu_single_wire_elementwise_enabled,
    _diagonal_region,
    _fused_gate_matrix,
    _gate_parameter_tensor,
    _StatevectorCrossWireDiagonalStep,
    _StatevectorCXSequenceStep,
    _StatevectorDisjointSingleWireStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorProgramStep,
    _StatevectorRXRZLoopStep,
    _StatevectorSingleWireRegion,
    _triton_parameterized_single_qubit_matrix_enabled,
    _triton_ry_rz_pair_enabled,
    _triton_single_qubit_loop_enabled,
    _triton_single_qubit_matrix_enabled,
)

if TYPE_CHECKING:
    from ...circuit import Circuit, _StatevectorExecutionStatistics


def _initial_state(circuit: Circuit) -> torch.Tensor:
    """Resolve or create a Circuit's local statevector input."""

    if circuit._inputs is not None:
        resolved = circuit._inputs.to(device=circuit.device, dtype=circuit.dtype)
        return resolved.reshape(1, -1) if resolved.ndim == 1 else resolved
    workspace = circuit._initial_state_workspace
    if workspace is None:
        workspace = torch.zeros(
            circuit.bsz,
            2**circuit.n_wires,
            dtype=circuit.dtype,
            device=circuit.device,
        )
        workspace[:, 0] = 1
        circuit._initial_state_workspace = workspace
    return workspace


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


def _rotation_region_angles(
    circuit: Circuit,
    steps: Sequence[_StatevectorFusedGateStep],
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    """Collect batched angles, or decline fusion when a parameter is unavailable."""
    regions: list[torch.Tensor] = []
    for step in steps:
        parameters = tuple(
            _gate_parameters(circuit, instruction, state, parameter_bindings)
            for instruction in step.instructions
        )
        angles: list[torch.Tensor] = []
        for parameter in parameters:
            if parameter is None:
                return None
            angles.append(parameter[:, 0])
        regions.append(torch.stack(angles, dim=-1))
    return torch.stack(regions, dim=0) if regions else None


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


def _initial_runtime_metrics(
    program: Sequence[_StatevectorProgramStep], *, enable_triton_loop: bool
) -> _StatevectorExecutionStatistics:
    metric_steps = tuple(
        region
        for step in program
        for region in (
            step.regions
            if isinstance(
                step,
                (_StatevectorCrossWireDiagonalStep, _StatevectorDisjointSingleWireStep),
            )
            else (step,)
        )
    )
    diagonal_metric_steps = tuple(
        region
        for step in program
        for region in (
            step.regions
            if isinstance(step, _StatevectorCrossWireDiagonalStep)
            else (step,)
        )
    )
    loop_steps = tuple(
        step for step in metric_steps if isinstance(step, _StatevectorRXRZLoopStep)
    )
    # Only flatten the dedicated diagonal wrapper for diagonal-path metrics.
    # A diagonal region inside a mixed single-wire group is materialized in the
    # group's dense Kronecker matrix and does not execute the elementwise path.
    # Counting it here would misreport which kernel actually ran.
    #
    # The predicate is the one the dispatch routes on, deliberately, including
    # its treatment of a supplied matrix: a gate named ``rz`` that carries a
    # matrix of its own does not take the diagonal kernel, so counting it here
    # would make the field disagree with the run it describes.
    diagonal_unfused_gates = sum(
        isinstance(step, _StatevectorGateStep) and _diagonal_region((step.instruction,))
        for step in diagonal_metric_steps
    )
    diagonal_fused_regions = tuple(
        step
        for step in diagonal_metric_steps
        if isinstance(step, _StatevectorFusedGateStep) and step.diagonal
    )
    return {
        "triton_single_qubit_loop_enabled": enable_triton_loop,
        "triton_single_qubit_loop_regions": len(loop_steps),
        "triton_single_qubit_loop_gates": sum(
            2 * len(step.pairs) for step in loop_steps
        ),
        "triton_ry_rz_pair_candidates": sum(
            isinstance(step, _StatevectorFusedGateStep)
            and tuple(item.name for item in step.instructions) == ("ry", "rz")
            for step in metric_steps
        ),
        "triton_ry_rz_pair_executed": 0,
        "triton_single_qubit_matrix_regions": 0,
        "diagonal_elementwise_gates": diagonal_unfused_gates
        + sum(len(step.instructions) for step in diagonal_fused_regions),
        "diagonal_fused_regions": len(diagonal_fused_regions),
        "permutation_gates": sum(
            isinstance(step, _StatevectorGateStep)
            and canonical_opcode(step.instruction.name) in {"x", "cx", "swap"}
            for step in program
        )
        + sum(
            len(step.controls)
            for step in metric_steps
            if isinstance(step, _StatevectorCXSequenceStep)
        ),
        "triton_cx_sequence_regions": sum(
            isinstance(step, _StatevectorCXSequenceStep) for step in metric_steps
        ),
        "fixed_single_qubit_specialized_gates": sum(
            isinstance(step, _StatevectorGateStep)
            and canonical_opcode(step.instruction.name) == "y"
            for step in program
        ),
        "fused_gate_regions": sum(
            isinstance(step, _StatevectorFusedGateStep) for step in metric_steps
        ),
        "fused_gate_count": sum(
            len(step.instructions)
            for step in metric_steps
            if isinstance(step, _StatevectorFusedGateStep)
        ),
        "dependency_reordered_single_qubit_regions": sum(
            isinstance(step, _StatevectorFusedGateStep) and step.dependency_reordered
            for step in metric_steps
        ),
        "statevector_apply_count": len(program),
    }


def _prepare_batched_rotation_matrices(
    circuit: Circuit,
    program: Sequence[_StatevectorProgramStep],
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    matrix_steps = tuple(
        region
        for step in program
        for region in (
            step.regions
            if isinstance(
                step,
                (_StatevectorCrossWireDiagonalStep, _StatevectorDisjointSingleWireStep),
            )
            else (step,)
        )
    )
    rx_ry_rz_steps = tuple(
        step
        for step in matrix_steps
        if isinstance(step, _StatevectorFusedGateStep)
        and tuple(item.name for item in step.instructions) == ("rx", "ry", "rz")
        and len(step.wires) == 1
    )
    rx_ry_rz_matrices: dict[int, torch.Tensor] = {}
    if rx_ry_rz_steps:
        region_angles = _rotation_region_angles(
            circuit, rx_ry_rz_steps, state, parameter_bindings
        )
        if region_angles is not None:
            matrices = _batched_rx_ry_rz_matrices(region_angles).to(dtype=state.dtype)
            rx_ry_rz_matrices = {
                id(step): matrices[index] for index, step in enumerate(rx_ry_rz_steps)
            }

    rotation_matrices: dict[int, torch.Tensor] = {}
    rotation_patterns = {
        tuple(item.name for item in step.instructions)
        for step in matrix_steps
        if isinstance(step, _StatevectorFusedGateStep)
        and len(step.instructions) >= 2
        and all(item.name in {"rx", "ry", "rz"} for item in step.instructions)
    }
    for names in rotation_patterns:
        rotation_steps = tuple(
            step
            for step in matrix_steps
            if isinstance(step, _StatevectorFusedGateStep)
            and tuple(item.name for item in step.instructions) == names
        )
        region_angles = _rotation_region_angles(
            circuit, rotation_steps, state, parameter_bindings
        )
        if region_angles is None:
            continue
        matrices = _batched_rotation_sequence_matrices(
            region_angles,
            names=names,
            dtype=state.dtype,
        )
        rotation_matrices.update(
            {id(step): matrices[index] for index, step in enumerate(rotation_steps)}
        )
    return rx_ry_rz_matrices, rotation_matrices


def _use_elementwise_single_wire(
    state: torch.Tensor,
    wires: Sequence[int],
) -> bool:
    """Whether the elementwise kernel is the one to apply this matrix with.

    Two tests, and neither is decorative. The kernel was measured on CPU only,
    and the CUDA path has its own single-qubit kernels, so the device test is
    what keeps an unmeasured device off the route; this host has no CUDA device,
    so no circuit can reach the branch it guards, and as a named predicate the
    condition is still an expression a test can evaluate on a ``meta`` tensor.
    The wire count is the kernel's whole domain: it combines the two amplitudes
    that differ in one wire, and it has no meaning for a gate spanning more.

    A diagonal one-wire matrix is in the domain too. The kernel is the general
    one-wire kernel, so a diagonal matrix is a case it handles, and on this host
    it is the faster of the two: at 20 wires a diagonal single-qubit region takes
    1.211 ms here against 1.392 ms through ``_apply_diagonal_matrix``.
    """

    return (
        len(wires) == 1
        and state.device.type == "cpu"
        and _cpu_single_wire_elementwise_enabled()
    )


def _apply_gate_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    *,
    uses_diagonal_kernel: bool,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Apply one already-built gate matrix with the kernel that fits it.

    ``_apply_matrix`` lays the state out, permutes the gate's wires to the front
    and permutes back, which a one-wire gate does not need.
    ``_apply_diagonal_matrix`` keeps the layout but replaces the batched matmul
    with an elementwise multiply, which only a diagonal matrix may take.

    ``uses_diagonal_kernel`` says the matrix is known to be diagonal, so the
    diagonal kernel is available here. It is not a preference: where the
    elementwise single-wire kernel applies as well, that kernel wins, because it
    drops the layout permutation the diagonal kernel still pays for, and it
    agrees with both of them to under an ulp on every shape measured. It is not
    bitwise equal to either, which is why the switch licensing it is off by
    default.
    """

    if _use_elementwise_single_wire(state, wires):
        return _apply_single_wire_matrix(state, matrix, wires[0], n_wires)
    apply_gate = _apply_diagonal_matrix if uses_diagonal_kernel else _apply_matrix
    return apply_gate(state, matrix, wires, n_wires, layout=layout)


def _apply_fused_gate_step(
    circuit: Circuit,
    step: _StatevectorFusedGateStep,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    rx_ry_rz_matrices: dict[int, torch.Tensor],
    rotation_matrices: dict[int, torch.Tensor],
) -> torch.Tensor:
    if (
        state.is_cuda
        and state.dtype == torch.complex64
        and _triton_ry_rz_pair_enabled()
        and tuple(item.name for item in step.instructions) == ("ry", "rz")
    ):
        ry_angles = _gate_parameters(
            circuit,
            step.instructions[0],
            state,
            parameter_bindings,
        )
        rz_angles = _gate_parameters(
            circuit,
            step.instructions[1],
            state,
            parameter_bindings,
        )
        if (
            ry_angles is not None
            and rz_angles is not None
            and not state.requires_grad
            and not ry_angles.requires_grad
            and not rz_angles.requires_grad
        ):
            from ..triton_kernels import ry_rz_pair

            circuit._last_statevector_runtime["triton_ry_rz_pair_executed"] += 1
            result: torch.Tensor = ry_rz_pair(
                state,
                ry_angles,
                rz_angles,
                wire=step.wires[0],
                n_wires=circuit.n_wires,
            )
            return result

    constant_matrix = _fused_constant_matrix(circuit, step, state, parameter_bindings)
    if (
        constant_matrix is not None
        and state.is_cuda
        and state.dtype == torch.complex64
        and _triton_single_qubit_matrix_enabled()
    ):
        from ..triton_kernels import single_qubit_matrix

        circuit._last_statevector_runtime["triton_single_qubit_matrix_regions"] += 1
        result = single_qubit_matrix(
            state,
            constant_matrix,
            wire=step.wires[0],
            n_wires=circuit.n_wires,
        )
        return result

    matrix = _fused_step_matrix(
        circuit,
        step,
        state,
        parameter_bindings,
        rx_ry_rz_matrices,
        rotation_matrices,
    )
    if (
        state.is_cuda
        and state.dtype == torch.complex64
        and len(step.wires) == 1
        and _triton_single_qubit_matrix_enabled()
        and (
            not matrix.requires_grad
            or _triton_parameterized_single_qubit_matrix_enabled()
        )
    ):
        from ..triton_kernels import single_qubit_matrix

        circuit._last_statevector_runtime["triton_single_qubit_matrix_regions"] += 1
        result = single_qubit_matrix(
            state,
            matrix,
            wire=step.wires[0],
            n_wires=circuit.n_wires,
        )
        return result
    return _apply_gate_matrix(
        state,
        matrix,
        step.wires,
        circuit.n_wires,
        uses_diagonal_kernel=step.diagonal,
        layout=step.layout,
    )


def _fused_step_matrix(
    circuit: Circuit,
    step: _StatevectorFusedGateStep,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    rx_ry_rz_matrices: dict[int, torch.Tensor],
    rotation_matrices: dict[int, torch.Tensor],
) -> torch.Tensor:
    matrix = rotation_matrices.get(id(step))
    if matrix is None:
        matrix = rx_ry_rz_matrices.get(id(step))
    if matrix is None:
        matrix = _fused_constant_matrix(circuit, step, state, parameter_bindings)
    if matrix is None:
        matrix = _fused_gate_matrix(
            step,
            bsz=state.shape[0],
            device=state.device,
            dtype=state.dtype,
            parameter_bindings=parameter_bindings,
        )
    return matrix


def _batched_kronecker_product(
    matrices: Sequence[torch.Tensor], *, batch_size: int
) -> torch.Tensor:
    """Build a batched Kronecker product without materializing batch pairs."""

    expanded: list[torch.Tensor] = []
    for matrix in matrices:
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.shape[0] == 1 and batch_size != 1:
            matrix = matrix.expand(batch_size, -1, -1)
        if matrix.shape != (batch_size, 2, 2):
            raise ValueError(
                "single-wire matrix must have shape [2, 2] or [batch, 2, 2]"
            )
        expanded.append(matrix)
    if not expanded:
        raise ValueError("at least one single-wire matrix is required")
    product = expanded[0]
    for matrix in expanded[1:]:
        rows, columns = product.shape[-2:]
        product = (product[:, :, None, :, None] * matrix[:, None, :, None, :]).reshape(
            batch_size, rows * 2, columns * 2
        )
    return product


def _apply_disjoint_single_wire_step(
    circuit: Circuit,
    step: _StatevectorDisjointSingleWireStep,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    rx_ry_rz_matrices: dict[int, torch.Tensor],
    rotation_matrices: dict[int, torch.Tensor],
) -> torch.Tensor:
    matrices = tuple(
        _single_wire_region_matrix(
            circuit,
            region,
            state,
            parameter_bindings,
            rx_ry_rz_matrices,
            rotation_matrices,
        )
        for region in step.regions
    )
    wires = tuple(
        (
            region.instruction.wires[0]
            if isinstance(region, _StatevectorGateStep)
            else region.wires[0]
        )
        for region in step.regions
    )
    matrix = _batched_kronecker_product(matrices, batch_size=state.shape[0])
    return _apply_matrix(state, matrix, wires, circuit.n_wires)


def _single_wire_region_matrix(
    circuit: Circuit,
    region: _StatevectorSingleWireRegion,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    rx_ry_rz_matrices: dict[int, torch.Tensor],
    rotation_matrices: dict[int, torch.Tensor],
) -> torch.Tensor:
    if isinstance(region, _StatevectorGateStep):
        return _gate_matrix(
            region.instruction,
            bsz=state.shape[0],
            device=state.device,
            dtype=state.dtype,
            parameter_bindings=parameter_bindings,
        )
    return _fused_step_matrix(
        circuit,
        region,
        state,
        parameter_bindings,
        rx_ry_rz_matrices,
        rotation_matrices,
    )


def _apply_cx_sequence(
    circuit: Circuit,
    step: _StatevectorCXSequenceStep,
    state: torch.Tensor,
) -> torch.Tensor:
    """Apply a compiled CX sequence through the available local kernel."""

    if state.is_cuda and state.dtype == torch.complex64:
        from ..triton_kernels import cx_sequence

        control_masks, target_masks, reverse_control_masks, reverse_target_masks = (
            _cx_sequence_masks(circuit, step, state.device)
        )
        return cx_sequence(
            state,
            control_masks=control_masks,
            target_masks=target_masks,
            reverse_control_masks=reverse_control_masks,
            reverse_target_masks=reverse_target_masks,
            n_wires=circuit.n_wires,
        )

    if (
        _cpu_cx_sequence_gather_enabled()
        and len(step.controls) >= _CX_SEQUENCE_GATHER_MINIMUM_LENGTH
    ):
        return _apply_cx_sequence_gather(
            state, step.controls, step.targets, circuit.n_wires
        )

    for control, target in zip(step.controls, step.targets, strict=True):
        state = _apply_cx_permutation(state, (control, target), circuit.n_wires)
    return state


def _apply_cross_wire_diagonal_step(
    circuit: Circuit,
    step: _StatevectorCrossWireDiagonalStep,
    state: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor:
    """Build wire-disjoint diagonals, then broadcast their product once."""

    wire_groups: list[tuple[int, ...]] = []
    diagonals: list[torch.Tensor] = []
    for region in step.regions:
        if isinstance(region, _StatevectorGateStep):
            instruction = region.instruction
            wires = tuple(instruction.wires)
            matrix = _gate_matrix(
                instruction,
                bsz=state.shape[0],
                device=state.device,
                dtype=state.dtype,
                parameter_bindings=parameter_bindings,
            )
        else:
            wires = region.wires
            constant_matrix = _fused_constant_matrix(
                circuit, region, state, parameter_bindings
            )
            if constant_matrix is None:
                matrix = _fused_gate_matrix(
                    region,
                    bsz=state.shape[0],
                    device=state.device,
                    dtype=state.dtype,
                    parameter_bindings=parameter_bindings,
                )
            else:
                matrix = constant_matrix
        wire_groups.append(wires)
        diagonals.append(torch.diagonal(matrix, dim1=-2, dim2=-1))
    return _apply_disjoint_diagonal_regions_cpu(
        state, diagonals, wire_groups, circuit.n_wires
    )


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
        enable_cpu_cross_wire_diagonal = (
            output.device.type == "cpu" and _cpu_cross_wire_diagonal_fusion_enabled()
        )
        enable_cpu_disjoint_single_wire = (
            output.device.type == "cpu"
            and _cpu_disjoint_single_wire_fusion_enabled()
            and not _cpu_single_wire_elementwise_enabled()
        )
        program_key = (
            "statevector",
            "triton_rx_rz_loop",
            enable_triton_loop,
            "cpu_cross_wire_diagonal",
            enable_cpu_cross_wire_diagonal,
            "cpu_disjoint_single_wire",
            enable_cpu_disjoint_single_wire,
        )
        program = circuit._backend_programs.get(program_key)
        if program is None:
            program = _compile_statevector_program(
                circuit._instructions,
                circuit.n_wires,
                enable_triton_loop=enable_triton_loop,
                enable_cpu_cross_wire_diagonal=enable_cpu_cross_wire_diagonal,
                enable_cpu_disjoint_single_wire=enable_cpu_disjoint_single_wire,
            )
            circuit._backend_programs[program_key] = program
        circuit._last_statevector_runtime = _initial_runtime_metrics(
            program, enable_triton_loop=enable_triton_loop
        )
        batched_rx_ry_rz_matrices, batched_rotation_matrices = (
            _prepare_batched_rotation_matrices(
                circuit,
                program,
                output,
                parameter_bindings,
            )
        )
        circuit._last_statevector_runtime["batched_rx_ry_rz_regions"] = len(
            batched_rx_ry_rz_matrices
        )
        circuit._last_statevector_runtime["batched_rotation_sequence_regions"] = len(
            batched_rotation_matrices
        )
        for step in program:
            if isinstance(step, _StatevectorDisjointSingleWireStep):
                output = _apply_disjoint_single_wire_step(
                    circuit,
                    step,
                    output,
                    parameter_bindings,
                    batched_rx_ry_rz_matrices,
                    batched_rotation_matrices,
                )
                continue
            if isinstance(step, _StatevectorCrossWireDiagonalStep):
                output = _apply_cross_wire_diagonal_step(
                    circuit,
                    step,
                    output,
                    parameter_bindings,
                )
                continue
            if isinstance(step, _StatevectorCXSequenceStep):
                output = _apply_cx_sequence(circuit, step, output)
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
                output = _apply_fused_gate_step(
                    circuit,
                    step,
                    output,
                    parameter_bindings,
                    batched_rx_ry_rz_matrices,
                    batched_rotation_matrices,
                )
                continue
            instruction = step.instruction
            name = canonical_opcode(instruction.name)
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
                output = _apply_gate_matrix(
                    output,
                    matrix,
                    instruction.wires,
                    circuit.n_wires,
                    uses_diagonal_kernel=_diagonal_region((instruction,)),
                    layout=step.layout,
                )
    circuit._state_cache = output
    return output


def _expectation_z(circuit: Circuit, wires: tuple[int, ...]) -> torch.Tensor:
    """Evaluate per-wire Z expectations for a local statevector circuit."""

    probabilities = torch.abs(state(circuit)) ** 2
    key = (wires, str(probabilities.device), probabilities.dtype)
    signs = circuit._statevector_z_signs.get(key)
    if signs is None:
        basis = torch.arange(
            probabilities.shape[-1],
            dtype=torch.int64,
            device=probabilities.device,
        )
        signs = torch.stack(
            tuple(
                1 - 2 * ((basis >> (circuit.n_wires - 1 - wire)) & 1) for wire in wires
            ),
            dim=-1,
        ).to(dtype=probabilities.dtype)
        circuit._statevector_z_signs[key] = signs
    return probabilities @ signs


def _expectation_pauli_string(
    circuit: Circuit,
    *,
    x: tuple[int, ...],
    y: tuple[int, ...],
    z: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate one X/Y/Z product observable for a local statevector circuit."""

    current_state = state(circuit)
    transformed = current_state
    for operator, wires in ((X_MATRIX, x), (Y_MATRIX, y), (Z_MATRIX, z)):
        matrix = operator.to(
            device=current_state.device,
            dtype=current_state.dtype,
        )
        for wire in wires:
            # Every term is a one-wire matrix, which is exactly what the
            # elementwise kernel is for: a full string costs one pass per wire.
            # The shipped code sent Z through ``_apply_matrix`` as well rather
            # than through the diagonal kernel, and that is kept.
            transformed = _apply_gate_matrix(
                transformed,
                matrix,
                (wire,),
                circuit.n_wires,
                uses_diagonal_kernel=False,
            )
    value = complex_mul(complex_conj(current_state), transformed).sum(dim=-1)
    return torch.real(value)


def _sample_statevector(
    circuit: Circuit,
    *,
    shots: int,
    generator: torch.Generator | None,
    return_bits: bool,
) -> torch.Tensor:
    """Sample local statevector probabilities as indices or bitstrings."""

    probabilities = torch.abs(state(circuit)) ** 2
    samples = torch.multinomial(
        probabilities,
        num_samples=shots,
        replacement=True,
        generator=generator,
    )
    return _bits_from_indices(samples, circuit.n_wires) if return_bits else samples


def _expectation_from_operators(
    *ops: tuple[Any, Sequence[int]],
    ket: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a dense operator product against an explicit statevector."""

    from ...circuit import Circuit

    input_state = ket.reshape(1, -1) if ket.ndim == 1 else ket
    n_wires = int(
        torch.log2(torch.tensor(input_state.shape[-1], dtype=torch.float32)).item()
    )
    circuit = Circuit(
        n_wires,
        bsz=input_state.shape[0],
        device=input_state.device,
        dtype=input_state.dtype if input_state.is_complex() else None,
        inputs=input_state,
    )
    for matrix, wires in ops:
        circuit.any(*wires, unitary=matrix)
    return complex_mul(complex_conj(input_state), state(circuit)).sum(dim=-1)


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

    from ...circuit import Circuit

    return Circuit.from_ir(
        program,
        bsz=batch_size,
        device=device,
        dtype=dtype,
    ).state()


__all__ = ("run_local_statevector", "state")
