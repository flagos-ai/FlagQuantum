"""Native PyTorch reverse mode over rank-local statevector shards."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from ....core.ir import CircuitIR, Instruction
from ....core.runtime_config import get_runtime_config, runtime_config
from ....simulation.statevector.adjoint import (
    analytic_rotation_derivative as _analytic_rotation_derivative,
)
from ....simulation.statevector.adjoint import (
    real_conjugate_inner_sum as _real_conjugate_inner_sum,
)
from ....simulation.statevector.adjoint import (
    z_expectation_adjoint_chunk as _z_expectation_adjoint_chunk,
)
from ....simulation.statevector.adjoint import (
    z_expectation_chunk as _z_expectation_chunk,
)
from ....simulation.statevector.operations import _instruction_matrix
from .forward import (
    StatevectorExchangeWorkspace,
    _is_diagonal_instruction,
    _storage_global_indices,
    _triton_local_cx_enabled,
    _triton_local_cx_segment_enabled,
    _vectorized_cross_shard_cx,
    _vectorized_local_cx_gate,
    _vectorized_local_diagonal_gate,
    _vectorized_local_gate,
    _vectorized_pair_exchange_gate,
    _vectorized_subgroup_exchange_gate,
    _wait_for_exchange,
)
from .gradient_reduction import AsyncGradientReducer
from .kernel_dispatch import triton_available
from .layout import (
    distributed_swap_rank_local_bits,
    plan_persistent_statevector_layout,
)
from .local_execution import (
    initialize_statevector_shard,
    use_compact_global_indices,
)
from .planning import plan_distributed_statevector
from .reverse import (
    BackwardExecutionEvidence,
    StatevectorCheckpointPolicy,
    _bind_parameters,
    _fused_vjp_pipeline_enabled,
    _gradient_bucketing_enabled,
    _gradient_reduction_overlap_enabled,
    _persistent_inplace_local_enabled,
    _persistent_wire_layout_enabled,
    _reverse_chunk_amplitudes,
    _reverse_cross_shard_cx_packing_enabled,
    _reverse_exchange_workspace_enabled,
    _triton_vjp_adjoint_decision,
)


def _compact_reverse_global_indices(plan: Any, rank: int) -> bool:
    """Use the shared shard-index representation policy for reverse states."""
    return bool(use_compact_global_indices(plan, rank))


def _local_expectation_z(
    shard_state: Any, *, plan: Any, n_wires: int, wire: int
) -> torch.Tensor:
    total = torch.zeros(
        (),
        dtype=shard_state.amplitudes.real.dtype,
        device=shard_state.amplitudes.device,
    )
    local_count = shard_state.shard.local_amplitudes
    for start in range(0, local_count, _reverse_chunk_amplitudes()):
        end = min(local_count, start + _reverse_chunk_amplitudes())
        indices = _storage_global_indices(shard_state, start, end, plan=plan)
        total = total + _z_expectation_chunk(
            shard_state.amplitudes[:, start:end],
            indices,
            n_wires=n_wires,
            wire=wire,
        )
    return total


def _local_expectation_z_adjoint(
    shard_state: Any, *, plan: Any, n_wires: int, wire: int
) -> torch.Tensor:
    """Construct d<Z>/d(state*) directly without an amplitude-sized graph."""

    adjoint = torch.empty_like(shard_state.amplitudes)
    local_count = shard_state.shard.local_amplitudes
    with torch.no_grad():
        for start in range(0, local_count, _reverse_chunk_amplitudes()):
            end = min(local_count, start + _reverse_chunk_amplitudes())
            indices = _storage_global_indices(shard_state, start, end, plan=plan)
            adjoint[:, start:end] = _z_expectation_adjoint_chunk(
                shard_state.amplitudes[:, start:end],
                indices,
                n_wires=n_wires,
                wire=wire,
            )
    return adjoint


def _apply_one_gate(
    shard_state: Any,
    *,
    instruction: Instruction,
    plan: Any,
    dtype: torch.dtype,
    process_group: Any | None = None,
    evidence: BackwardExecutionEvidence | None = None,
    workspace: StatevectorExchangeWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> Any:
    precision = get_runtime_config().with_overrides(
        complex_dtype=str(dtype).removeprefix("torch.")
    )
    with runtime_config(precision):
        matrix = _instruction_matrix(
            instruction, device=shard_state.amplitudes.device, dtype=dtype
        )
    return _apply_matrix_gate(
        shard_state,
        matrix=matrix,
        instruction=instruction,
        plan=plan,
        process_group=process_group,
        evidence=evidence,
        workspace=workspace,
        output=output,
    )


def _apply_matrix_gate(
    shard_state: Any,
    *,
    matrix: torch.Tensor,
    instruction: Instruction,
    plan: Any,
    process_group: Any | None = None,
    evidence: BackwardExecutionEvidence | None = None,
    workspace: StatevectorExchangeWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> Any:
    if _is_diagonal_instruction(instruction.name):
        state, scratch = _vectorized_local_diagonal_gate(
            shard_state,
            matrix,
            instruction.wires,
            plan=plan,
            chunk_amplitudes=_reverse_chunk_amplitudes(),
            output=output,
        )
        if evidence is not None:
            evidence.peak_scratch_bytes = max(evidence.peak_scratch_bytes, scratch)
        return state
    touched = any(wire in plan.sharded_wires for wire in instruction.wires)
    if not touched or plan.world_size == 1:
        if (
            _triton_local_cx_enabled()
            and instruction.name == "cx"
            and shard_state.amplitudes.device.type == "cuda"
            and shard_state.amplitudes.dtype == torch.complex64
        ):
            state, scratch = _vectorized_local_cx_gate(
                shard_state,
                instruction.wires,
                plan=plan,
                output=output,
            )
        else:
            state, scratch = _vectorized_local_gate(
                shard_state,
                matrix,
                instruction.wires,
                plan=plan,
                chunk_amplitudes=_reverse_chunk_amplitudes(),
                output=output,
            )
        if evidence is not None:
            evidence.peak_scratch_bytes = max(evidence.peak_scratch_bytes, scratch)
        return state
    if (
        _reverse_cross_shard_cx_packing_enabled()
        and instruction.name == "cx"
        and sum(wire in plan.sharded_wires for wire in instruction.wires) == 1
    ):
        state, count, byte_count, scratch = _vectorized_cross_shard_cx(
            shard_state,
            instruction.wires,
            plan=plan,
            chunk_amplitudes=_reverse_chunk_amplitudes(),
            process_group=process_group,
            workspace=workspace,
            output=output,
        )
    elif len(instruction.wires) == 1 and instruction.wires[0] in plan.sharded_wires:
        state, count, byte_count, scratch = _vectorized_pair_exchange_gate(
            shard_state,
            matrix,
            instruction.wires[0],
            plan=plan,
            chunk_amplitudes=_reverse_chunk_amplitudes(),
            process_group=process_group,
            workspace=workspace,
            output=output,
        )
    else:
        state, count, byte_count, scratch = _vectorized_subgroup_exchange_gate(
            shard_state,
            matrix,
            instruction.wires,
            plan=plan,
            chunk_amplitudes=_reverse_chunk_amplitudes(),
            process_group=process_group,
            workspace=workspace,
            output=output,
        )
    if evidence is not None:
        evidence.communication_count += count
        evidence.communication_bytes += byte_count
        evidence.peak_scratch_bytes = max(evidence.peak_scratch_bytes, scratch)
    return state


def _fused_sharded_1q_vjp_adjoint(
    before: Any,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    derivative_matrix: torch.Tensor,
    *,
    wire: int,
    plan: Any,
    process_group: Any | None,
    workspace: StatevectorExchangeWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Exchange before/adjoint chunks once and fuse the sharded 1q reverse work."""

    from ....simulation.triton_kernels.statevector_adjoint import (
        fused_complex64_sharded_1q_vjp_adjoint,
    )

    position = plan.sharded_wires.index(int(wire))
    rank_shift = len(plan.sharded_wires) - position - 1
    peer = before.rank ^ (1 << rank_shift)
    global_peer = (
        dist.get_global_rank(process_group, peer) if process_group is not None else peer
    )
    rank_basis = (before.rank >> rank_shift) & 1
    next_adjoint = torch.empty_like(adjoint) if output is None else output
    local_count = before.shard.local_amplitudes
    gradient = torch.zeros((), dtype=torch.float32, device=adjoint.device)
    communication_count = communication_bytes = peak = 0
    output_bytes = next_adjoint.numel() * next_adjoint.element_size()
    chunks = tuple(
        (start, min(local_count, start + _reverse_chunk_amplitudes()))
        for start in range(0, local_count, _reverse_chunk_amplitudes())
    )

    def post(chunk_index: int) -> tuple[Any, ...]:
        start, end = chunks[chunk_index]
        local_before = before.amplitudes[:, start:end].contiguous()
        local_adjoint = adjoint[:, start:end].contiguous()
        slot_base = 2 * (chunk_index % 2)
        remote_before = (
            workspace.acquire(local_before, slot=slot_base)
            if workspace is not None
            else torch.empty_like(local_before)
        )
        remote_adjoint = (
            workspace.acquire(local_adjoint, slot=slot_base + 1)
            if workspace is not None
            else torch.empty_like(local_adjoint)
        )
        operations = [
            dist.P2POp(
                dist.isend,
                local_before,
                global_peer,
                process_group,
                2 * chunk_index,
            ),
            dist.P2POp(
                dist.irecv,
                remote_before,
                global_peer,
                process_group,
                2 * chunk_index,
            ),
            dist.P2POp(
                dist.isend,
                local_adjoint,
                global_peer,
                process_group,
                2 * chunk_index + 1,
            ),
            dist.P2POp(
                dist.irecv,
                remote_adjoint,
                global_peer,
                process_group,
                2 * chunk_index + 1,
            ),
        ]
        requests = tuple(dist.batch_isend_irecv(operations))
        return (
            start,
            end,
            local_before,
            local_adjoint,
            remote_before,
            remote_adjoint,
            requests,
        )

    current = post(0)
    pipelined = (
        workspace is not None and len(chunks) > 1 and _fused_vjp_pipeline_enabled()
    )
    if pipelined:
        workspace.pipelined_gate_count += 1
    for chunk_index in range(len(chunks)):
        (
            start,
            end,
            local_before,
            local_adjoint,
            remote_before,
            remote_adjoint,
            requests,
        ) = current
        for request in requests:
            _wait_for_exchange(request, adjoint.device)
        following = None
        if pipelined and chunk_index + 1 < len(chunks):
            following = post(chunk_index + 1)
            workspace.pipeline_prefetch_count += 1
        updated, partial_gradient = fused_complex64_sharded_1q_vjp_adjoint(
            local_before,
            remote_before,
            local_adjoint,
            remote_adjoint,
            matrix,
            derivative_matrix,
            rank_basis=rank_basis,
        )
        next_adjoint[:, start:end] = updated
        gradient += partial_gradient
        exchanged_bytes = 2 * local_before.numel() * local_before.element_size()
        communication_count += 2
        communication_bytes += exchanged_bytes
        if workspace is not None:
            workspace.record_peer_exchange(global_peer, exchanged_bytes)
        chunk_bytes = local_before.numel() * local_before.element_size()
        pipeline_factor = 4 if following is not None else 2
        peak = max(peak, output_bytes + (3 + pipeline_factor) * chunk_bytes)
        if chunk_index + 1 < len(chunks):
            current = following if following is not None else post(chunk_index + 1)
    return next_adjoint, gradient, communication_count, communication_bytes, peak


def _explicit_sharded_adjoint(
    ir: CircuitIR,
    slots: tuple[tuple[int, str, int], ...],
    saved_parameters: tuple[torch.Tensor, ...],
    *,
    observable_wire: int,
    device: torch.device,
    policy: StatevectorCheckpointPolicy,
    process_group: Any | None = None,
    evidence: BackwardExecutionEvidence,
    saved_final_state: Any | None = None,
    saved_inter_node_ket_checkpoints: tuple[tuple[int, int, torch.Tensor], ...] = (),
) -> tuple[torch.Tensor, ...]:
    """Rematerialize rank-local forward states and propagate gate adjoints."""

    world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
    rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    dtype = (
        torch.complex128
        if any(p.dtype == torch.float64 for p in saved_parameters)
        else torch.complex64
    )
    base_parameters = tuple(value.detach().to(device) for value in saved_parameters)
    bound = _bind_parameters(ir, slots, base_parameters)
    plan = plan_distributed_statevector(
        bound,
        world_size=world_size,
        local_world_size=world_size,
        bsz=1,
        complex_bytes=torch.empty((), dtype=dtype).element_size(),
    )
    exchange_workspace = (
        StatevectorExchangeWorkspace()
        if _reverse_exchange_workspace_enabled()
        else None
    )
    persistent_layout = (
        _persistent_wire_layout_enabled()
        and world_size > 1
        and policy.strategy == "reversible_adjoint"
    )
    persistent_plan = (
        plan_persistent_statevector_layout(bound, world_size=world_size)
        if persistent_layout
        else None
    )
    evidence.persistent_layout_enabled = persistent_plan is not None
    persistent_mapping = list(
        persistent_plan.final_logical_to_physical
        if persistent_plan is not None
        else range(ir.n_wires)
    )
    swaps_before: dict[int, list[Any]] = {}
    if persistent_plan is not None:
        for swap in persistent_plan.swaps:
            swaps_before.setdefault(swap.before_instruction, []).append(swap)
    inter_node_ket_checkpoints = {
        (index, swap_index): amplitudes
        for index, swap_index, amplitudes in saved_inter_node_ket_checkpoints
    }
    local_wire_capacity = ir.n_wires - plan.rank_address_bits
    inplace_local = bool(
        (persistent_plan is not None or world_size == 1)
        and _persistent_inplace_local_enabled()
        and all(
            len(instruction.wires) <= local_wire_capacity
            for instruction in bound.instructions
        )
    )
    evidence.persistent_inplace_local_enabled = inplace_local
    layout_send_buffer = layout_receive_buffer = None
    if persistent_plan is not None and persistent_plan.swaps:
        local_amplitudes = plan.shards[rank].local_amplitudes
        half_shape = (plan.bsz, local_amplitudes // 2)
        layout_send_buffer = torch.empty(half_shape, dtype=dtype, device=device)
        layout_receive_buffer = torch.empty_like(layout_send_buffer)
        evidence.persistent_layout_buffer_allocation_count = 2
        evidence.persistent_layout_buffer_reserved_bytes = (
            2 * layout_send_buffer.numel() * layout_send_buffer.element_size()
        )

    initial = None
    checkpoints: dict[int, Any] = {}
    if saved_final_state is None:
        initial = initialize_statevector_shard(
            plan,
            rank=rank,
            device=device,
            dtype=dtype,
            compact_global_indices=_compact_reverse_global_indices(plan, rank),
        )
        checkpoints[0] = initial
    if policy.strategy == "interval":
        assert initial is not None
        state = initial
        with torch.no_grad():
            for index in range(len(bound.instructions)):
                state = _apply_one_gate(
                    state,
                    instruction=bound.instructions[index],
                    plan=plan,
                    dtype=dtype,
                    process_group=process_group,
                    evidence=evidence,
                    workspace=exchange_workspace,
                )
                if (index + 1) % policy.interval == 0:
                    checkpoints[index + 1] = replace(
                        state, amplitudes=state.amplitudes.detach().clone()
                    )

    def state_before(stop: int) -> Any:
        start = max(index for index in checkpoints if index <= stop)
        checkpoint = checkpoints[start]
        # Forward kernels are out-of-place, so a checkpoint can safely be the
        # replay input.  Cloning here retained another complete rank-local
        # state for every adjoint gate (8 GiB at 31q/world-size 2).
        state = replace(checkpoint, amplitudes=checkpoint.amplitudes.detach())
        with torch.no_grad():
            for index in range(start, stop):
                state = _apply_one_gate(
                    state,
                    instruction=bound.instructions[index],
                    plan=plan,
                    dtype=dtype,
                    process_group=process_group,
                    evidence=evidence,
                    workspace=exchange_workspace,
                )
        return state

    final_state = (
        state_before(len(bound.instructions))
        if saved_final_state is None
        else replace(
            saved_final_state,
            amplitudes=saved_final_state.amplitudes.detach(),
        )
    )
    evidence.saved_forward_state_reused = saved_final_state is not None
    final_observable_wire = persistent_mapping[observable_wire]
    adjoint = _local_expectation_z_adjoint(
        final_state, plan=plan, n_wires=ir.n_wires, wire=final_observable_wire
    )
    reversible_state = final_state if policy.strategy == "reversible_adjoint" else None
    if reversible_state is None:
        del final_state
        reversible_scratch = None
        adjoint_scratch = None
    else:
        # A reversible sweep reconstructs every preceding ket from the final
        # ket with U†, so retaining the zero-state replay checkpoint wastes one
        # complete shard (16 GiB for a 31q complex64 single-GPU run).
        checkpoints.clear()
        if initial is not None:
            del initial
        del final_state
        # Allocate the only two full-shard work buffers up front.  Every
        # subsequent inverse ket/adjoint gate alternates between these fixed
        # allocations instead of asking the CUDA allocator for another 16 GiB.
        reversible_scratch = (
            None if inplace_local else torch.empty_like(reversible_state.amplitudes)
        )
        adjoint_scratch = None if inplace_local else torch.empty_like(adjoint)
        if os.getenv("FQ_STATEVECTOR_DEBUG_MEMORY", "0") == "1":
            torch.cuda.synchronize(device)
            print(
                "reversible_adjoint_initialized",
                {
                    "allocated": torch.cuda.memory_allocated(device),
                    "reserved": torch.cuda.memory_reserved(device),
                    "shard_bytes": reversible_state.amplitudes.numel()
                    * reversible_state.amplitudes.element_size(),
                },
                flush=True,
            )
    accumulated = [torch.zeros_like(parameter) for parameter in base_parameters]
    slots_by_instruction: dict[int, set[int]] = {}
    for instruction_index, _, parameter_index in slots:
        slots_by_instruction.setdefault(instruction_index, set()).add(parameter_index)
    remaining_parameter_instructions = [0] * len(base_parameters)
    for parameter_indices in slots_by_instruction.values():
        for parameter_index in parameter_indices:
            remaining_parameter_instructions[parameter_index] += 1
    gradient_reducer = (
        AsyncGradientReducer(
            accumulated,
            process_group=process_group,
            evidence=evidence,
            max_parameters=(None if _gradient_bucketing_enabled() else 1),
            max_bytes=(None if _gradient_bucketing_enabled() else 1),
            async_op=_gradient_reduction_overlap_enabled(),
        )
        if world_size > 1
        else None
    )

    def accumulate_parameter_gradient(
        parameter_index: int, gradient: torch.Tensor
    ) -> None:
        accumulated[parameter_index] += gradient.detach()
        evidence.reduction_counts[parameter_index] += 1
        remaining_parameter_instructions[parameter_index] -= 1
        if (
            gradient_reducer is not None
            and remaining_parameter_instructions[parameter_index] == 0
        ):
            gradient_reducer.mark_ready(
                parameter_index,
                overlap_opportunity=index > 0,
            )

    skipped_cx_indices: set[int] = set()
    cx_segment_ket_scratch = cx_segment_adjoint_scratch = None
    for index in range(len(bound.instructions) - 1, -1, -1):
        if index in skipped_cx_indices:
            continue
        execution_instruction = replace(
            bound.instructions[index],
            wires=tuple(
                persistent_mapping[int(wire)]
                for wire in bound.instructions[index].wires
            ),
        )
        active = sorted(slots_by_instruction.get(index, ()))
        reversible_vjp_decision = _triton_vjp_adjoint_decision(
            supported=bool(
                inplace_local
                and reversible_state is not None
                and len(active) == 1
                and len(execution_instruction.wires) == 1
                and execution_instruction.wires[0] not in plan.sharded_wires
                and reversible_state.amplitudes.dtype == torch.complex64
                and reversible_state.amplitudes.device.type == "cuda"
            )
        )
        if reversible_vjp_decision.accelerated:
            precision = get_runtime_config().with_overrides(
                complex_dtype=str(dtype).removeprefix("torch.")
            )
            with runtime_config(precision):
                original_matrix = _instruction_matrix(
                    execution_instruction, device=device, dtype=dtype
                )
            derivative_matrix = _analytic_rotation_derivative(
                execution_instruction, original_matrix
            )
            if derivative_matrix is not None:
                evidence.kernel_dispatch_evidence.record(reversible_vjp_decision)
                from ....simulation.triton_kernels.statevector_adjoint import (
                    fused_complex64_local_1q_reversible_vjp,
                )

                bit_position = (
                    plan.n_wires
                    - execution_instruction.wires[0]
                    - 1
                    - len(plan.sharded_wires)
                )
                gradient = fused_complex64_local_1q_reversible_vjp(
                    reversible_state.amplitudes,
                    adjoint,
                    original_matrix,
                    derivative_matrix,
                    bit_position=bit_position,
                ).to(dtype=base_parameters[active[0]].dtype)
                accumulate_parameter_gradient(active[0], gradient)
                evidence.analytic_rotation_derivative_count += 1
                evidence.fused_parameter_adjoint_count += 1
                for swap_index, swap in reversed(
                    tuple(enumerate(swaps_before.get(index, ())))
                ):
                    checkpoint = inter_node_ket_checkpoints.pop(
                        (index, swap_index), None
                    )
                    if checkpoint is None:
                        _, count, byte_count = distributed_swap_rank_local_bits(
                            reversible_state.amplitudes,
                            rank=rank,
                            n_wires=ir.n_wires,
                            rank_bits=plan.rank_address_bits,
                            local_physical_wire=swap.local_physical_wire,
                            sharded_physical_wire=swap.sharded_physical_wire,
                            process_group=process_group,
                            output=reversible_state.amplitudes,
                            send_buffer=layout_send_buffer,
                            receive_buffer=layout_receive_buffer,
                            tag=(index * max(1, plan.rank_address_bits) + swap_index)
                            * 2,
                        )
                    else:
                        reversible_state = replace(
                            reversible_state, amplitudes=checkpoint
                        )
                        count = byte_count = 0
                    _, adjoint_count, adjoint_bytes = distributed_swap_rank_local_bits(
                        adjoint,
                        rank=rank,
                        n_wires=ir.n_wires,
                        rank_bits=plan.rank_address_bits,
                        local_physical_wire=swap.local_physical_wire,
                        sharded_physical_wire=swap.sharded_physical_wire,
                        process_group=process_group,
                        output=adjoint,
                        send_buffer=layout_send_buffer,
                        receive_buffer=layout_receive_buffer,
                        tag=(index * max(1, plan.rank_address_bits) + swap_index) * 2
                        + 1,
                    )
                    evidence.communication_count += count + adjoint_count
                    evidence.communication_bytes += byte_count + adjoint_bytes
                    evidence.persistent_layout_swap_count += 1
                    evidence.persistent_layout_swap_bytes += byte_count + adjoint_bytes
                    (
                        persistent_mapping[swap.local_logical_wire],
                        persistent_mapping[swap.sharded_logical_wire],
                    ) = (
                        persistent_mapping[swap.sharded_logical_wire],
                        persistent_mapping[swap.local_logical_wire],
                    )
                continue
        if (
            inplace_local
            and triton_available()
            and _triton_local_cx_segment_enabled(bound)
            and execution_instruction.name == "cx"
            and not swaps_before.get(index)
        ):
            segment_start = index
            while segment_start > 0:
                candidate_index = segment_start - 1
                candidate = bound.instructions[candidate_index]
                if candidate.name != "cx" or swaps_before.get(candidate_index):
                    break
                segment_start = candidate_index
            if segment_start < index:
                from ....simulation.triton_kernels.statevector_gates import (
                    apply_complex64_local_cx_segment,
                )

                segment_wires = tuple(
                    tuple(persistent_mapping[int(wire)] for wire in item.wires)
                    for item in bound.instructions[segment_start : index + 1]
                )
                if not any(
                    wire in plan.sharded_wires
                    for wires in segment_wires
                    for wire in wires
                ):
                    rank_bits = len(plan.sharded_wires)
                    controls = tuple(
                        plan.n_wires - wires[0] - 1 - rank_bits
                        for wires in reversed(segment_wires)
                    )
                    targets = tuple(
                        plan.n_wires - wires[1] - 1 - rank_bits
                        for wires in reversed(segment_wires)
                    )
                    if cx_segment_ket_scratch is None:
                        cx_segment_ket_scratch = torch.empty_like(
                            reversible_state.amplitudes
                        )
                        cx_segment_adjoint_scratch = torch.empty_like(adjoint)
                    previous_ket = reversible_state.amplitudes
                    apply_complex64_local_cx_segment(
                        previous_ket,
                        control_bit_positions=controls,
                        target_bit_positions=targets,
                        output=cx_segment_ket_scratch,
                    )
                    reversible_state = replace(
                        reversible_state, amplitudes=cx_segment_ket_scratch
                    )
                    cx_segment_ket_scratch = previous_ket
                    previous_adjoint = adjoint
                    apply_complex64_local_cx_segment(
                        previous_adjoint,
                        control_bit_positions=controls,
                        target_bit_positions=targets,
                        output=cx_segment_adjoint_scratch,
                    )
                    adjoint = cx_segment_adjoint_scratch
                    cx_segment_adjoint_scratch = previous_adjoint
                    evidence.peak_scratch_bytes = max(
                        evidence.peak_scratch_bytes,
                        2
                        * reversible_state.amplitudes.numel()
                        * reversible_state.amplitudes.element_size(),
                    )
                    skipped_cx_indices.update(range(segment_start, index))
                    continue
        original_matrix: torch.Tensor | None = None
        if reversible_state is None:
            before = state_before(index)
        else:
            precision = get_runtime_config().with_overrides(
                complex_dtype=str(dtype).removeprefix("torch.")
            )
            with runtime_config(precision):
                inverse_matrix = _instruction_matrix(
                    execution_instruction, device=device, dtype=dtype
                ).mH
            original_matrix = inverse_matrix.mH
            previous_reversible_state = reversible_state
            previous_reversible_amplitudes = previous_reversible_state.amplitudes
            inverse_output = (
                previous_reversible_amplitudes if inplace_local else reversible_scratch
            )
            before = _apply_matrix_gate(
                replace(
                    previous_reversible_state,
                    amplitudes=previous_reversible_state.amplitudes.detach(),
                ),
                matrix=inverse_matrix,
                instruction=execution_instruction,
                plan=plan,
                process_group=process_group,
                evidence=evidence,
                workspace=exchange_workspace,
                output=inverse_output,
            )
            reversible_state = before
            if not inplace_local:
                reversible_scratch = previous_reversible_amplitudes
            del previous_reversible_state, inverse_matrix
        fused_adjoint: torch.Tensor | None = None
        if active:
            instruction = execution_instruction
            if original_matrix is None:
                precision = get_runtime_config().with_overrides(
                    complex_dtype=str(dtype).removeprefix("torch.")
                )
                with runtime_config(precision):
                    original_matrix = _instruction_matrix(
                        instruction, device=device, dtype=dtype
                    )
            derivatives = []
            for parameter_index in active:
                parameter = base_parameters[parameter_index]
                if parameter.numel() != 1:
                    raise ValueError(
                        "sharded statevector reverse currently expects scalar "
                        "gate parameters"
                    )

                def matrix_for_parameter(value: torch.Tensor) -> torch.Tensor:
                    values = list(base_parameters)
                    values[parameter_index] = value
                    current_bound = _bind_parameters(ir, slots, values)
                    precision = get_runtime_config().with_overrides(
                        complex_dtype=str(dtype).removeprefix("torch.")
                    )
                    with runtime_config(precision):
                        return _instruction_matrix(
                            current_bound.instructions[index],
                            device=device,
                            dtype=dtype,
                        )

                derivative_matrix = _analytic_rotation_derivative(
                    instruction,
                    original_matrix,
                )
                if derivative_matrix is not None:
                    evidence.analytic_rotation_derivative_count += 1
                if derivative_matrix is None:
                    # Generic gates retain the tiny-matrix JVP fallback.
                    _, derivative_matrix = torch.autograd.functional.jvp(
                        matrix_for_parameter,
                        (parameter.detach().requires_grad_(True),),
                        (torch.ones_like(parameter),),
                        create_graph=False,
                        strict=False,
                    )
                vjp_decision = _triton_vjp_adjoint_decision(
                    supported=bool(
                        len(active) == 1
                        and len(instruction.wires) == 1
                        and before.amplitudes.dtype == torch.complex64
                        and before.amplitudes.device.type == "cuda"
                    )
                )
                evidence.kernel_dispatch_evidence.record(vjp_decision)
                if vjp_decision.accelerated:
                    from ....simulation.triton_kernels.statevector_adjoint import (
                        fused_complex64_local_1q_vjp_adjoint,
                    )

                    wire = instruction.wires[0]
                    if wire in plan.sharded_wires and plan.world_size > 1:
                        (
                            fused_adjoint,
                            local_derivative,
                            communication_count,
                            communication_bytes,
                            scratch_bytes,
                        ) = _fused_sharded_1q_vjp_adjoint(
                            before,
                            adjoint,
                            original_matrix,
                            derivative_matrix,
                            wire=wire,
                            plan=plan,
                            process_group=process_group,
                            workspace=exchange_workspace,
                            output=adjoint if inplace_local else adjoint_scratch,
                        )
                        evidence.communication_count += communication_count
                        evidence.communication_bytes += communication_bytes
                        evidence.peak_scratch_bytes = max(
                            evidence.peak_scratch_bytes, scratch_bytes
                        )
                    else:
                        bit_position = plan.n_wires - wire - 1 - len(plan.sharded_wires)
                        fused_adjoint, local_derivative = (
                            fused_complex64_local_1q_vjp_adjoint(
                                before.amplitudes,
                                adjoint,
                                original_matrix,
                                derivative_matrix,
                                bit_position=bit_position,
                                output=adjoint if inplace_local else adjoint_scratch,
                            )
                        )
                        evidence.peak_scratch_bytes = max(
                            evidence.peak_scratch_bytes,
                            fused_adjoint.numel() * fused_adjoint.element_size(),
                        )
                    local_derivative = local_derivative.to(dtype=parameter.dtype)
                    derivatives.append(local_derivative)
                    evidence.fused_parameter_adjoint_count += 1
                    del derivative_matrix
                    continue
                with torch.no_grad():
                    derivative_state = _apply_matrix_gate(
                        replace(before, amplitudes=before.amplitudes.detach()),
                        matrix=derivative_matrix,
                        instruction=execution_instruction,
                        plan=plan,
                        process_group=process_group,
                        evidence=evidence,
                        workspace=exchange_workspace,
                    )
                    local_derivative = torch.zeros_like(parameter)
                    local_count = derivative_state.shard.local_amplitudes
                    for start in range(0, local_count, _reverse_chunk_amplitudes()):
                        end = min(local_count, start + _reverse_chunk_amplitudes())
                        local_derivative += _real_conjugate_inner_sum(
                            adjoint[:, start:end],
                            derivative_state.amplitudes[:, start:end],
                        ).to(dtype=parameter.dtype)
                derivatives.append(local_derivative)
                del derivative_state, derivative_matrix
        else:
            derivatives = ()
        for parameter_index, gradient in zip(active, derivatives):
            if gradient is None:
                continue
            accumulate_parameter_gradient(parameter_index, gradient)
        if fused_adjoint is not None:
            previous_adjoint = adjoint
            adjoint = fused_adjoint
            if not inplace_local and adjoint_scratch is not None:
                adjoint_scratch = previous_adjoint
        else:
            if original_matrix is None:
                precision = get_runtime_config().with_overrides(
                    complex_dtype=str(dtype).removeprefix("torch.")
                )
                with runtime_config(precision):
                    original_matrix = _instruction_matrix(
                        bound.instructions[index], device=device, dtype=dtype
                    )
            adjoint_state = replace(before, amplitudes=adjoint.detach())
            previous_adjoint = adjoint
            adjoint = _apply_matrix_gate(
                adjoint_state,
                matrix=original_matrix.mH,
                instruction=execution_instruction,
                plan=plan,
                process_group=process_group,
                evidence=evidence,
                workspace=exchange_workspace,
                output=adjoint if inplace_local else adjoint_scratch,
            ).amplitudes
            if not inplace_local and adjoint_scratch is not None:
                adjoint_scratch = previous_adjoint
        for swap_index, swap in reversed(tuple(enumerate(swaps_before.get(index, ())))):
            assert reversible_state is not None
            previous_reversible = reversible_state.amplitudes
            checkpoint = inter_node_ket_checkpoints.pop((index, swap_index), None)
            if checkpoint is None:
                restored_reversible, count, byte_count = (
                    distributed_swap_rank_local_bits(
                        previous_reversible,
                        rank=rank,
                        n_wires=ir.n_wires,
                        rank_bits=plan.rank_address_bits,
                        local_physical_wire=swap.local_physical_wire,
                        sharded_physical_wire=swap.sharded_physical_wire,
                        process_group=process_group,
                        output=(
                            previous_reversible if inplace_local else reversible_scratch
                        ),
                        send_buffer=layout_send_buffer,
                        receive_buffer=layout_receive_buffer,
                        tag=(index * max(1, plan.rank_address_bits) + swap_index) * 2,
                    )
                )
            else:
                restored_reversible = checkpoint
                count = byte_count = 0
            reversible_state = replace(reversible_state, amplitudes=restored_reversible)
            if not inplace_local:
                reversible_scratch = previous_reversible
            previous_adjoint = adjoint
            adjoint, adjoint_count, adjoint_bytes = distributed_swap_rank_local_bits(
                previous_adjoint,
                rank=rank,
                n_wires=ir.n_wires,
                rank_bits=plan.rank_address_bits,
                local_physical_wire=swap.local_physical_wire,
                sharded_physical_wire=swap.sharded_physical_wire,
                process_group=process_group,
                output=previous_adjoint if inplace_local else adjoint_scratch,
                send_buffer=layout_send_buffer,
                receive_buffer=layout_receive_buffer,
                tag=(index * max(1, plan.rank_address_bits) + swap_index) * 2 + 1,
            )
            if not inplace_local:
                adjoint_scratch = previous_adjoint
            evidence.communication_count += count + adjoint_count
            evidence.communication_bytes += byte_count + adjoint_bytes
            evidence.persistent_layout_swap_count += 1
            evidence.persistent_layout_swap_bytes += byte_count + adjoint_bytes
            (
                persistent_mapping[swap.local_logical_wire],
                persistent_mapping[swap.sharded_logical_wire],
            ) = (
                persistent_mapping[swap.sharded_logical_wire],
                persistent_mapping[swap.local_logical_wire],
            )
        if reversible_state is None:
            del before
    if gradient_reducer is not None:
        gradient_reducer.finish()
    if exchange_workspace is not None:
        evidence.exchange_workspace_allocation_count = (
            exchange_workspace.allocation_count
        )
        evidence.exchange_workspace_reuse_count = exchange_workspace.reuse_count
        evidence.exchange_workspace_reserved_bytes = exchange_workspace.reserved_bytes
        evidence.exchange_pipeline_prefetch_count = (
            exchange_workspace.pipeline_prefetch_count
        )
        evidence.intra_node_communication_count = (
            exchange_workspace.intra_node_message_count
        )
        evidence.intra_node_communication_bytes = exchange_workspace.intra_node_bytes
        evidence.inter_node_communication_count = (
            exchange_workspace.inter_node_message_count
        )
        evidence.inter_node_communication_bytes = exchange_workspace.inter_node_bytes
    return tuple(accumulated)
