"""Torch-distributed statevector forward execution loop."""

from __future__ import annotations

import os
import warnings
from dataclasses import replace
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....core.ir import ensure_circuit_ir
from ....core.runtime_config import get_runtime_config, runtime_config
from ....simulation.statevector_ops import _compose_gate_matrices, _instruction_matrix
from ...distributed.flagos_runtime import current_flagos_device
from ...distributed.identity import build_distributed_identity
from .forward import (
    StatevectorExchangeWorkspace,
    TorchDistributedStatevectorResult,
    _cross_shard_cx_packing_enabled,
    _is_diagonal_instruction,
    _ket_checkpoint_mode,
    _local_block_fusion_enabled,
    _local_block_fusion_width,
    _triton_local_cx_decision,
    _triton_local_cx_segment_enabled,
    _triton_transpose_1q_enabled,
    _vectorized_cross_shard_cx,
    _vectorized_local_cx_gate,
    _vectorized_local_diagonal_gate,
    _vectorized_local_gate,
    _vectorized_pair_exchange_gate,
    _vectorized_subgroup_exchange_gate,
    communication_aware_wire_layout,
)
from .kernel_dispatch import KernelDispatchEvidence, triton_available
from .layout import (
    distributed_swap_rank_local_bits,
    plan_persistent_statevector_layout,
    schedule_statevector_dependency_dag,
)
from .local_execution import (
    initialize_statevector_shard,
    use_compact_global_indices,
)
from .planning import plan_distributed_statevector


def execute_torch_distributed_statevector(
    circuit_or_ir: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
    local_world_size: int | None = None,
    exchange_buffer_bytes: int = 512 * 1024 * 1024,
    compact_index_threshold: int = 1 << 24,
    process_group: Any | None = None,
    fuse_cross_shard_gates: bool = True,
    pipeline_pair_exchange: bool = True,
    wire_layout: str = "canonical",
    preferred_local_wires: Sequence[int] = (),
    persistent_wire_layout: bool = False,
) -> TorchDistributedStatevectorResult:
    """Execute validated IR while retaining only the current rank's shard."""

    ir = ensure_circuit_ir(circuit_or_ir)
    world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
    rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    backend = (
        dist.get_backend(process_group) if dist.is_initialized() else "single_process"
    )
    if world_size > 1 and os.getenv(
        "FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT", "1"
    ).strip().lower() in {"0", "false", "off", "no"}:
        warnings.warn(
            "FlagQuantum distributed statevector is running in baseline mode; "
            "set FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT=1 (the default) to enable "
            "persistent layout and dependency scheduling.",
            RuntimeWarning,
            stacklevel=2,
        )
    if wire_layout not in {"canonical", "communication_aware"}:
        raise ValueError("wire_layout must be 'canonical' or 'communication_aware'")
    logical_to_physical = tuple(range(ir.n_wires))
    if wire_layout == "communication_aware":
        ir, logical_to_physical = communication_aware_wire_layout(
            ir,
            world_size=world_size,
            preferred_local_wires=preferred_local_wires,
        )
    if persistent_wire_layout and world_size > 1:
        ir = schedule_statevector_dependency_dag(ir, world_size=world_size)
    persistent_plan = (
        plan_persistent_statevector_layout(ir, world_size=world_size)
        if persistent_wire_layout and world_size > 1
        else None
    )
    persistent_mapping = list(range(ir.n_wires))
    local_compilation = persistent_plan is not None or world_size == 1
    swaps_before: dict[int, list[Any]] = {}
    if persistent_plan is not None:
        for swap in persistent_plan.swaps:
            swaps_before.setdefault(swap.before_instruction, []).append(swap)
        # Cross-shard fusion groups are expressed in the initial layout.
        fuse_cross_shard_gates = False
    if device is None:
        if backend == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        elif backend == "flagos":
            device = current_flagos_device()
        else:
            device = torch.device("cpu")
    resolved_device = torch.device(device)
    if backend == "nccl" and resolved_device.type != "cuda":
        raise ValueError("NCCL execution requires a CUDA device")
    if backend == "flagos" and resolved_device.type != "flagos":
        raise ValueError("FlagOS execution requires a flagos device")
    if local_world_size is None:
        if process_group is not None:
            # Torchrun topology variables describe the default world and cannot
            # safely describe an arbitrary subgroup.
            local_world_size = world_size
        else:
            local_world_size = int(
                os.environ.get("LOCAL_WORLD_SIZE")
                or os.environ.get("NPROC_PER_NODE")
                or world_size
            )
    plan = plan_distributed_statevector(
        ir,
        world_size=world_size,
        local_world_size=local_world_size,
        bsz=1,
        complex_bytes=torch.empty((), dtype=dtype).element_size(),
    )
    validation = plan.validate()
    if not validation.valid:
        raise ValueError(
            "invalid distributed statevector plan: " + "; ".join(validation.errors)
        )
    precision = get_runtime_config().with_overrides(
        complex_dtype=str(dtype).removeprefix("torch.")
    )
    with runtime_config(precision):
        matrices = tuple(
            _instruction_matrix(item, device=resolved_device, dtype=dtype)
            for item in ir.instructions
        )
    exchange_workspace = (
        StatevectorExchangeWorkspace()
        if not any(matrix.requires_grad for matrix in matrices)
        else None
    )
    chunk_amplitudes = max(
        1,
        exchange_buffer_bytes
        // (plan.bsz * torch.empty((), dtype=dtype).element_size() * 4),
    )
    shard_state = initialize_statevector_shard(
        plan,
        rank=rank,
        device=resolved_device,
        dtype=dtype,
        compact_global_indices=use_compact_global_indices(
            plan,
            rank,
            local_amplitude_threshold=compact_index_threshold,
        ),
    )
    layout_send_buffer = layout_receive_buffer = None
    if persistent_plan is not None and persistent_plan.swaps:
        half_shape = (
            shard_state.amplitudes.shape[0],
            shard_state.amplitudes.shape[1] // 2,
        )
        layout_send_buffer = torch.empty(
            half_shape, dtype=dtype, device=resolved_device
        )
        layout_receive_buffer = torch.empty_like(layout_send_buffer)
    cx_segment_scratch = (
        torch.empty_like(shard_state.amplitudes)
        if (
            local_compilation
            and triton_available()
            and _triton_local_cx_segment_enabled(ir)
            and resolved_device.type == "cuda"
            and dtype == torch.complex64
        )
        else None
    )
    local_count = distributed_count = communication_count = communication_bytes = 0
    kernel_dispatch_evidence = KernelDispatchEvidence()
    local_diagonal_count = 0
    executed_distributed_segments = 0
    fused_cross_shard_regions = 0
    fused_cross_shard_gate_count = 0
    peak_scratch = 0
    persistent_inter_node_ket_checkpoints: list[tuple[int, int, torch.Tensor]] = []
    local_rank_bits = int(local_world_size).bit_length() - 1
    index = 0
    while index < len(ir.instructions):
        instruction_swaps = swaps_before.get(index, ())
        instruction = ir.instructions[index]
        fused_swap_gate = bool(
            _triton_transpose_1q_enabled()
            and local_compilation
            and not torch.is_grad_enabled()
            and len(instruction_swaps) == 1
            and len(instruction.wires) == 1
            and instruction_swaps[0].sharded_logical_wire == int(instruction.wires[0])
            and shard_state.amplitudes.dtype == torch.complex64
            and shard_state.amplitudes.device.type == "cuda"
        )
        for swap_index, swap in enumerate(instruction_swaps):
            rank_bit_position = ir.n_wires - swap.sharded_physical_wire - 1
            checkpoint_mode = _ket_checkpoint_mode()
            if checkpoint_mode == "all" or (
                checkpoint_mode == "inter_node"
                and world_size > local_world_size
                and rank_bit_position >= local_rank_bits
            ):
                persistent_inter_node_ket_checkpoints.append(
                    (
                        index,
                        swap_index,
                        shard_state.amplitudes.detach().clone(),
                    )
                )
            swapped, count, byte_count = distributed_swap_rank_local_bits(
                shard_state.amplitudes,
                rank=rank,
                n_wires=ir.n_wires,
                rank_bits=plan.rank_address_bits,
                local_physical_wire=swap.local_physical_wire,
                sharded_physical_wire=swap.sharded_physical_wire,
                process_group=process_group,
                output=shard_state.amplitudes,
                send_buffer=layout_send_buffer,
                receive_buffer=layout_receive_buffer,
                fused_matrix=matrices[index] if fused_swap_gate else None,
                tag=index * max(1, plan.rank_address_bits) + swap_index,
            )
            shard_state = replace(shard_state, amplitudes=swapped)
            communication_count += count
            communication_bytes += byte_count
            (
                persistent_mapping[swap.local_logical_wire],
                persistent_mapping[swap.sharded_logical_wire],
            ) = (
                persistent_mapping[swap.sharded_logical_wire],
                persistent_mapping[swap.local_logical_wire],
            )
            peak_scratch = max(
                peak_scratch,
                swapped.numel() * swapped.element_size() + 2 * byte_count,
            )
        if fused_swap_gate:
            local_count += 1
            index += 1
            continue
        original_instruction = ir.instructions[index]
        instruction = replace(
            original_instruction,
            wires=tuple(
                persistent_mapping[int(wire)] for wire in original_instruction.wires
            ),
        )
        matrix = matrices[index]
        touched = any(wire in plan.sharded_wires for wire in instruction.wires)
        if cx_segment_scratch is not None and instruction.name == "cx" and not touched:
            segment_wires = [instruction.wires]
            segment_cursor = index + 1
            while segment_cursor < len(ir.instructions):
                if swaps_before.get(segment_cursor):
                    break
                candidate = ir.instructions[segment_cursor]
                if candidate.name != "cx":
                    break
                mapped = tuple(
                    persistent_mapping[int(wire)] for wire in candidate.wires
                )
                if any(wire in plan.sharded_wires for wire in mapped):
                    break
                segment_wires.append(mapped)
                segment_cursor += 1
            if len(segment_wires) > 1:
                from ....simulation.triton_kernels.statevector_gates import (
                    apply_complex64_local_cx_segment,
                )

                rank_bits = len(plan.sharded_wires)
                controls = tuple(
                    plan.n_wires - wires[0] - 1 - rank_bits for wires in segment_wires
                )
                targets = tuple(
                    plan.n_wires - wires[1] - 1 - rank_bits for wires in segment_wires
                )
                previous = shard_state.amplitudes
                apply_complex64_local_cx_segment(
                    previous,
                    control_bit_positions=controls,
                    target_bit_positions=targets,
                    output=cx_segment_scratch,
                )
                shard_state = replace(shard_state, amplitudes=cx_segment_scratch)
                cx_segment_scratch = previous
                local_count += len(segment_wires)
                peak_scratch = max(
                    peak_scratch,
                    shard_state.amplitudes.numel()
                    * shard_state.amplitudes.element_size(),
                )
                index = segment_cursor
                continue
        if (
            local_compilation
            and _local_block_fusion_enabled()
            and len(instruction.wires) == 1
            and not touched
        ):
            block_instructions = [instruction]
            block_matrices = [matrix]
            block_wires = [instruction.wires[0]]
            block_cursor = index + 1
            while block_cursor < min(
                index + _local_block_fusion_width(), len(ir.instructions)
            ):
                if swaps_before.get(block_cursor):
                    break
                candidate_original = ir.instructions[block_cursor]
                if len(candidate_original.wires) != 1:
                    break
                candidate_wire = persistent_mapping[int(candidate_original.wires[0])]
                if (
                    candidate_wire in plan.sharded_wires
                    or candidate_wire in block_wires
                ):
                    break
                block_instructions.append(
                    replace(candidate_original, wires=(candidate_wire,))
                )
                block_matrices.append(matrices[block_cursor])
                block_wires.append(candidate_wire)
                block_cursor += 1
            if len(block_instructions) > 1:
                combined = block_matrices[0]
                for candidate_matrix in block_matrices[1:]:
                    combined = torch.kron(combined, candidate_matrix)
                shard_state, scratch = _vectorized_local_gate(
                    shard_state,
                    combined,
                    tuple(block_wires),
                    plan=plan,
                    chunk_amplitudes=chunk_amplitudes,
                    output=shard_state.amplitudes,
                    kernel_dispatch_evidence=kernel_dispatch_evidence,
                )
                peak_scratch = max(peak_scratch, scratch)
                local_count += len(block_instructions)
                index = block_cursor
                continue
        cursor = index + 1
        if touched:
            while cursor < len(ir.instructions):
                candidate = ir.instructions[cursor]
                if tuple(candidate.wires) != tuple(instruction.wires):
                    break
                cursor += 1
        if fuse_cross_shard_gates and touched and cursor - index > 1:
            group = ir.instructions[index:cursor]
            matrix = _compose_gate_matrices(matrices[index:cursor])
            diagonal_count = sum(_is_diagonal_instruction(item.name) for item in group)
            non_diagonal_count = len(group) - diagonal_count
            if non_diagonal_count == 0:
                shard_state, scratch = _vectorized_local_diagonal_gate(
                    shard_state,
                    matrix,
                    instruction.wires,
                    plan=plan,
                    chunk_amplitudes=chunk_amplitudes,
                )
                local_count += len(group)
                local_diagonal_count += len(group)
            elif len(instruction.wires) == 1:
                shard_state, count, byte_count, scratch = (
                    _vectorized_pair_exchange_gate(
                        shard_state,
                        matrix,
                        instruction.wires[0],
                        plan=plan,
                        chunk_amplitudes=chunk_amplitudes,
                        process_group=process_group,
                        workspace=exchange_workspace,
                        pipeline=pipeline_pair_exchange,
                    )
                )
                local_count += diagonal_count
                local_diagonal_count += diagonal_count
                distributed_count += non_diagonal_count
                executed_distributed_segments += 1
                communication_count += count
                communication_bytes += byte_count
                fused_cross_shard_regions += 1
                fused_cross_shard_gate_count += len(group)
            else:
                shard_state, count, byte_count, scratch = (
                    _vectorized_subgroup_exchange_gate(
                        shard_state,
                        matrix,
                        instruction.wires,
                        plan=plan,
                        chunk_amplitudes=chunk_amplitudes,
                        process_group=process_group,
                        workspace=exchange_workspace,
                        pipeline=pipeline_pair_exchange,
                    )
                )
                local_count += diagonal_count
                local_diagonal_count += diagonal_count
                distributed_count += non_diagonal_count
                executed_distributed_segments += 1
                communication_count += count
                communication_bytes += byte_count
                fused_cross_shard_regions += 1
                fused_cross_shard_gate_count += len(group)
            peak_scratch = max(peak_scratch, scratch)
            index = cursor
            continue
        if _is_diagonal_instruction(instruction.name):
            shard_state, scratch = _vectorized_local_diagonal_gate(
                shard_state,
                matrix,
                instruction.wires,
                plan=plan,
                chunk_amplitudes=chunk_amplitudes,
                output=(shard_state.amplitudes if local_compilation else None),
            )
            peak_scratch = max(peak_scratch, scratch)
            local_count += 1
            local_diagonal_count += 1
            index += 1
            continue
        if not touched or world_size == 1:
            local_output = shard_state.amplitudes if local_compilation else None
            cx_decision = _triton_local_cx_decision(
                supported=bool(
                    instruction.name == "cx"
                    and shard_state.amplitudes.device.type == "cuda"
                    and shard_state.amplitudes.dtype == torch.complex64
                )
            )
            if instruction.name == "cx":
                kernel_dispatch_evidence.record(cx_decision)
            if cx_decision.accelerated:
                shard_state, scratch = _vectorized_local_cx_gate(
                    shard_state,
                    instruction.wires,
                    plan=plan,
                    output=local_output,
                )
            else:
                shard_state, scratch = _vectorized_local_gate(
                    shard_state,
                    matrix,
                    instruction.wires,
                    plan=plan,
                    chunk_amplitudes=chunk_amplitudes,
                    output=local_output,
                    kernel_dispatch_evidence=kernel_dispatch_evidence,
                )
            peak_scratch = max(peak_scratch, scratch)
            local_count += 1
            index += 1
            continue
        if (
            _cross_shard_cx_packing_enabled()
            and instruction.name == "cx"
            and sum(wire in plan.sharded_wires for wire in instruction.wires) == 1
        ):
            shard_state, count, byte_count, scratch = _vectorized_cross_shard_cx(
                shard_state,
                instruction.wires,
                plan=plan,
                chunk_amplitudes=chunk_amplitudes,
                process_group=process_group,
                workspace=exchange_workspace,
            )
        elif len(instruction.wires) == 1 and instruction.wires[0] in plan.sharded_wires:
            shard_state, count, byte_count, scratch = _vectorized_pair_exchange_gate(
                shard_state,
                matrix,
                instruction.wires[0],
                plan=plan,
                chunk_amplitudes=chunk_amplitudes,
                process_group=process_group,
                workspace=exchange_workspace,
                pipeline=pipeline_pair_exchange,
            )
        else:
            shard_state, count, byte_count, scratch = (
                _vectorized_subgroup_exchange_gate(
                    shard_state,
                    matrix,
                    instruction.wires,
                    plan=plan,
                    chunk_amplitudes=chunk_amplitudes,
                    process_group=process_group,
                    workspace=exchange_workspace,
                    pipeline=pipeline_pair_exchange,
                )
            )
        distributed_count += 1
        executed_distributed_segments += 1
        communication_count += count
        communication_bytes += byte_count
        peak_scratch = max(peak_scratch, scratch)
        index += 1
    if persistent_plan is not None:
        logical_to_physical = tuple(
            persistent_mapping[logical_to_physical[wire]] for wire in range(ir.n_wires)
        )
        wire_layout = "persistent"
    return TorchDistributedStatevectorResult(
        shard_state=shard_state,
        plan=plan,
        backend=str(backend),
        local_gate_count=local_count,
        distributed_gate_count=distributed_count,
        communication_count=communication_count,
        communication_bytes=communication_bytes,
        peak_scratch_bytes=peak_scratch,
        local_diagonal_gate_count=local_diagonal_count,
        executed_distributed_segment_count=executed_distributed_segments,
        fused_cross_shard_regions=fused_cross_shard_regions,
        fused_cross_shard_gate_count=fused_cross_shard_gate_count,
        cross_shard_fusion_enabled=bool(fuse_cross_shard_gates),
        exchange_workspace_allocation_count=(
            exchange_workspace.allocation_count if exchange_workspace is not None else 0
        ),
        exchange_workspace_reuse_count=(
            exchange_workspace.reuse_count if exchange_workspace is not None else 0
        ),
        exchange_workspace_reserved_bytes=(
            exchange_workspace.reserved_bytes if exchange_workspace is not None else 0
        ),
        exchange_pipeline_prefetch_count=(
            exchange_workspace.pipeline_prefetch_count
            if exchange_workspace is not None
            else 0
        ),
        exchange_pipelined_gate_count=(
            exchange_workspace.pipelined_gate_count
            if exchange_workspace is not None
            else 0
        ),
        exchange_pipeline_enabled=bool(pipeline_pair_exchange),
        exchange_subgroup_pipelined_gate_count=(
            exchange_workspace.subgroup_pipelined_gate_count
            if exchange_workspace is not None
            else 0
        ),
        intra_node_communication_count=(
            exchange_workspace.intra_node_message_count
            if exchange_workspace is not None
            else 0
        ),
        intra_node_communication_bytes=(
            exchange_workspace.intra_node_bytes if exchange_workspace is not None else 0
        ),
        inter_node_communication_count=(
            exchange_workspace.inter_node_message_count
            if exchange_workspace is not None
            else 0
        ),
        inter_node_communication_bytes=(
            exchange_workspace.inter_node_bytes if exchange_workspace is not None else 0
        ),
        wire_layout=wire_layout,
        logical_to_physical_wires=logical_to_physical,
        distributed_identity=build_distributed_identity(
            outer_backend=str(backend),
            logical_device=str(resolved_device),
            rank=rank,
            world_size=world_size,
            process_group_initialized=dist.is_initialized(),
        ),
        kernel_dispatch_evidence=kernel_dispatch_evidence,
        persistent_inter_node_ket_checkpoints=tuple(
            persistent_inter_node_ket_checkpoints
        ),
    )
