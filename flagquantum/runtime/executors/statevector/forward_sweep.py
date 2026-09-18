"""One-instruction-at-a-time dispatch over a sharded statevector."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from ....compute import resolve_platform_device
from ....core.ir import Instruction, ensure_circuit_ir
from ....core.runtime_config import get_runtime_config, runtime_config
from ....simulation.statevector.operations import (
    _compose_gate_matrices,
    _instruction_matrix,
)
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
from .local_execution import initialize_statevector_shard, use_compact_global_indices
from .planning import plan_distributed_statevector


class _ShardedForwardSweep:
    """Execute a validated IR, keeping only this rank's shard."""

    def __init__(
        self,
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
    ) -> None:
        """Resolve the plan, the layout, and every buffer the sweep needs."""

        self.device = device
        self.dtype = dtype
        self.exchange_buffer_bytes = exchange_buffer_bytes
        self.compact_index_threshold = compact_index_threshold
        self.process_group = process_group
        self.fuse_cross_shard_gates = fuse_cross_shard_gates
        self.pipeline_pair_exchange = pipeline_pair_exchange
        self.wire_layout = wire_layout
        self.preferred_local_wires = preferred_local_wires
        self.persistent_wire_layout = persistent_wire_layout

        self.ir = ensure_circuit_ir(circuit_or_ir)
        self.world_size = (
            dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        )
        self.rank = dist.get_rank(self.process_group) if dist.is_initialized() else 0
        self.backend = (
            dist.get_backend(self.process_group)
            if dist.is_initialized()
            else "single_process"
        )
        if self.world_size > 1 and os.getenv(
            "FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT", "1"
        ).strip().lower() in {"0", "false", "off", "no"}:
            warnings.warn(
                "FlagQuantum distributed statevector is running in baseline mode; "
                "set FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT=1 (the default) to enable "
                "persistent layout and dependency scheduling.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.wire_layout not in {"canonical", "communication_aware"}:
            raise ValueError("wire_layout must be 'canonical' or 'communication_aware'")
        self.logical_to_physical = tuple(range(self.ir.n_wires))
        if self.wire_layout == "communication_aware":
            self.ir, self.logical_to_physical = communication_aware_wire_layout(
                self.ir,
                world_size=self.world_size,
                preferred_local_wires=self.preferred_local_wires,
            )
        if self.persistent_wire_layout and self.world_size > 1:
            self.ir = schedule_statevector_dependency_dag(
                self.ir, world_size=self.world_size
            )
        self.persistent_plan = (
            plan_persistent_statevector_layout(self.ir, world_size=self.world_size)
            if self.persistent_wire_layout and self.world_size > 1
            else None
        )
        self.persistent_mapping = list(range(self.ir.n_wires))
        self.local_compilation = (
            self.persistent_plan is not None or self.world_size == 1
        )
        self.swaps_before: dict[int, list[Any]] = {}
        if self.persistent_plan is not None:
            for swap in self.persistent_plan.swaps:
                self.swaps_before.setdefault(swap.before_instruction, []).append(swap)
            # Cross-shard fusion groups are expressed in the initial layout.
            self.fuse_cross_shard_gates = False
        if self.device is None:
            if self.backend == "nccl":
                self.device = resolve_platform_device("cuda")
            elif self.backend == "flagos":
                self.device = current_flagos_device()
            else:
                self.device = torch.device("cpu")
        self.resolved_device = torch.device(self.device)
        if self.backend == "nccl" and self.resolved_device.type != "cuda":
            raise ValueError("NCCL execution requires a CUDA device")
        if self.backend == "flagos" and self.resolved_device.type != "flagos":
            raise ValueError("FlagOS execution requires a flagos device")
        if local_world_size is None:
            if self.process_group is not None:
                # Torchrun topology variables describe the default world and cannot
                # safely describe an arbitrary subgroup.
                local_world_size = self.world_size
            else:
                local_world_size = int(
                    os.environ.get("LOCAL_WORLD_SIZE")
                    or os.environ.get("NPROC_PER_NODE")
                    or self.world_size
                )
        self.local_world_size = local_world_size
        self.plan = plan_distributed_statevector(
            self.ir,
            world_size=self.world_size,
            local_world_size=local_world_size,
            bsz=1,
            complex_bytes=torch.empty((), dtype=self.dtype).element_size(),
        )
        self.validation = self.plan.validate()
        if not self.validation.valid:
            raise ValueError(
                "invalid distributed statevector plan: "
                + "; ".join(self.validation.errors)
            )
        self.precision = get_runtime_config().with_overrides(
            complex_dtype=str(self.dtype).removeprefix("torch.")
        )
        with runtime_config(self.precision):
            self.matrices = tuple(
                _instruction_matrix(item, device=self.resolved_device, dtype=self.dtype)
                for item in self.ir.instructions
            )
        self.exchange_workspace = (
            StatevectorExchangeWorkspace()
            if not any(matrix.requires_grad for matrix in self.matrices)
            else None
        )
        self.chunk_amplitudes = max(
            1,
            self.exchange_buffer_bytes
            // (self.plan.bsz * torch.empty((), dtype=self.dtype).element_size() * 4),
        )
        self.shard_state = initialize_statevector_shard(
            self.plan,
            rank=self.rank,
            device=self.resolved_device,
            dtype=self.dtype,
            compact_global_indices=use_compact_global_indices(
                self.plan,
                self.rank,
                local_amplitude_threshold=self.compact_index_threshold,
            ),
        )
        self.layout_send_buffer = self.layout_receive_buffer = None
        if self.persistent_plan is not None and self.persistent_plan.swaps:
            self.half_shape = (
                self.shard_state.amplitudes.shape[0],
                self.shard_state.amplitudes.shape[1] // 2,
            )
            self.layout_send_buffer = torch.empty(
                self.half_shape, dtype=self.dtype, device=self.resolved_device
            )
            self.layout_receive_buffer = torch.empty_like(self.layout_send_buffer)
        self.cx_segment_scratch = (
            torch.empty_like(self.shard_state.amplitudes)
            if (
                self.local_compilation
                and triton_available()
                and _triton_local_cx_segment_enabled(self.ir)
                and self.resolved_device.type == "cuda"
                and self.dtype == torch.complex64
            )
            else None
        )
        self.local_count = self.distributed_count = self.communication_count = (
            self.communication_bytes
        ) = 0
        self.kernel_dispatch_evidence = KernelDispatchEvidence()
        self.local_diagonal_count = 0
        self.executed_distributed_segments = 0
        self.fused_cross_shard_regions = 0
        self.fused_cross_shard_gate_count = 0
        self.peak_scratch = 0
        self.persistent_inter_node_ket_checkpoints: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self.local_rank_bits = int(local_world_size).bit_length() - 1
        self.index = 0

    def run(self) -> TorchDistributedStatevectorResult:
        """Execute the instruction stream and build the result."""

        while self.index < len(self.ir.instructions):
            self.index = self.step(self.index)
        return self._result()

    def step(self, index: int) -> int:
        """Run one instruction's dispatch step and return the index to resume at."""

        instruction_swaps = self.swaps_before.get(index, ())
        instruction = self.ir.instructions[index]
        fused_swap_gate = bool(
            _triton_transpose_1q_enabled()
            and self.local_compilation
            and not torch.is_grad_enabled()
            and len(instruction_swaps) == 1
            and len(instruction.wires) == 1
            and instruction_swaps[0].sharded_logical_wire == int(instruction.wires[0])
            and self.shard_state.amplitudes.dtype == torch.complex64
            and self.shard_state.amplitudes.device.type == "cuda"
        )
        for swap_index, swap in enumerate(instruction_swaps):
            rank_bit_position = self.ir.n_wires - swap.sharded_physical_wire - 1
            checkpoint_mode = _ket_checkpoint_mode()
            if checkpoint_mode == "all" or (
                checkpoint_mode == "inter_node"
                and self.world_size > self.local_world_size
                and rank_bit_position >= self.local_rank_bits
            ):
                self.persistent_inter_node_ket_checkpoints.append(
                    (
                        index,
                        swap_index,
                        self.shard_state.amplitudes.detach().clone(),
                    )
                )
            swapped, count, byte_count = distributed_swap_rank_local_bits(
                self.shard_state.amplitudes,
                rank=self.rank,
                n_wires=self.ir.n_wires,
                rank_bits=self.plan.rank_address_bits,
                local_physical_wire=swap.local_physical_wire,
                sharded_physical_wire=swap.sharded_physical_wire,
                process_group=self.process_group,
                output=self.shard_state.amplitudes,
                send_buffer=self.layout_send_buffer,
                receive_buffer=self.layout_receive_buffer,
                fused_matrix=self.matrices[index] if fused_swap_gate else None,
                tag=index * max(1, self.plan.rank_address_bits) + swap_index,
            )
            self.shard_state = replace(self.shard_state, amplitudes=swapped)
            self.communication_count += count
            self.communication_bytes += byte_count
            (
                self.persistent_mapping[swap.local_logical_wire],
                self.persistent_mapping[swap.sharded_logical_wire],
            ) = (
                self.persistent_mapping[swap.sharded_logical_wire],
                self.persistent_mapping[swap.local_logical_wire],
            )
            self.peak_scratch = max(
                self.peak_scratch,
                swapped.numel() * swapped.element_size() + 2 * byte_count,
            )
        if fused_swap_gate:
            self.local_count += 1
            return index + 1
        original_instruction = self.ir.instructions[index]
        instruction = replace(
            original_instruction,
            wires=tuple(
                self.persistent_mapping[int(wire)]
                for wire in original_instruction.wires
            ),
        )
        matrix = self.matrices[index]
        touched = any(wire in self.plan.sharded_wires for wire in instruction.wires)
        cursor = self._cx_segment(index, instruction, touched)
        if cursor is not None:
            return cursor
        cursor = self._local_block_fusion(index, instruction, matrix, touched)
        if cursor is not None:
            return cursor
        cursor = index + 1
        if touched:
            while cursor < len(self.ir.instructions):
                candidate = self.ir.instructions[cursor]
                if tuple(candidate.wires) != tuple(instruction.wires):
                    break
                cursor += 1
        if self._cross_shard_fusion(index, instruction, matrix, touched, cursor):
            return cursor
        if self._diagonal(index, instruction, matrix):
            return index + 1
        if self._local(index, instruction, matrix, touched):
            return index + 1
        self._cross_shard(index, instruction, matrix)
        return index + 1

    def _cx_segment(
        self,
        index: int,
        instruction: Instruction,
        touched: bool,
    ) -> int | None:
        """Fuse a contiguous run of local CX gates, or report it cannot."""

        if (
            self.cx_segment_scratch is not None
            and instruction.name == "cx"
            and not touched
        ):
            segment_wires = [instruction.wires]
            segment_cursor = index + 1
            while segment_cursor < len(self.ir.instructions):
                if self.swaps_before.get(segment_cursor):
                    break
                candidate = self.ir.instructions[segment_cursor]
                if candidate.name != "cx":
                    break
                mapped = tuple(
                    self.persistent_mapping[int(wire)] for wire in candidate.wires
                )
                if any(wire in self.plan.sharded_wires for wire in mapped):
                    break
                segment_wires.append(mapped)
                segment_cursor += 1
            if len(segment_wires) > 1:
                from ....simulation.triton_kernels.statevector_gates import (
                    apply_complex64_local_cx_segment,
                )

                rank_bits = len(self.plan.sharded_wires)
                controls = tuple(
                    self.plan.n_wires - wires[0] - 1 - rank_bits
                    for wires in segment_wires
                )
                targets = tuple(
                    self.plan.n_wires - wires[1] - 1 - rank_bits
                    for wires in segment_wires
                )
                previous = self.shard_state.amplitudes
                apply_complex64_local_cx_segment(
                    previous,
                    control_bit_positions=controls,
                    target_bit_positions=targets,
                    output=self.cx_segment_scratch,
                )
                self.shard_state = replace(
                    self.shard_state, amplitudes=self.cx_segment_scratch
                )
                self.cx_segment_scratch = previous
                self.local_count += len(segment_wires)
                self.peak_scratch = max(
                    self.peak_scratch,
                    self.shard_state.amplitudes.numel()
                    * self.shard_state.amplitudes.element_size(),
                )
                return segment_cursor
        return None

    def _local_block_fusion(
        self,
        index: int,
        instruction: Instruction,
        matrix: torch.Tensor,
        touched: bool,
    ) -> int | None:
        """Fuse a run of adjacent local one-qubit gates, or report it cannot."""

        if (
            self.local_compilation
            and _local_block_fusion_enabled()
            and len(instruction.wires) == 1
            and not touched
        ):
            block_instructions = [instruction]
            block_matrices = [matrix]
            block_wires = [instruction.wires[0]]
            block_cursor = index + 1
            while block_cursor < min(
                index + _local_block_fusion_width(), len(self.ir.instructions)
            ):
                if self.swaps_before.get(block_cursor):
                    break
                candidate_original = self.ir.instructions[block_cursor]
                if len(candidate_original.wires) != 1:
                    break
                candidate_wire = self.persistent_mapping[
                    int(candidate_original.wires[0])
                ]
                if (
                    candidate_wire in self.plan.sharded_wires
                    or candidate_wire in block_wires
                ):
                    break
                block_instructions.append(
                    replace(candidate_original, wires=(candidate_wire,))
                )
                block_matrices.append(self.matrices[block_cursor])
                block_wires.append(candidate_wire)
                block_cursor += 1
            if len(block_instructions) > 1:
                combined = block_matrices[0]
                for candidate_matrix in block_matrices[1:]:
                    combined = torch.kron(combined, candidate_matrix)
                self.shard_state, scratch = _vectorized_local_gate(
                    self.shard_state,
                    combined,
                    tuple(block_wires),
                    plan=self.plan,
                    chunk_amplitudes=self.chunk_amplitudes,
                    output=self.shard_state.amplitudes,
                    kernel_dispatch_evidence=self.kernel_dispatch_evidence,
                )
                self.peak_scratch = max(self.peak_scratch, scratch)
                self.local_count += len(block_instructions)
                return block_cursor
        return None

    def _cross_shard_fusion(
        self,
        index: int,
        instruction: Instruction,
        matrix: torch.Tensor,
        touched: bool,
        cursor: int,
    ) -> bool:
        """Apply a fused cross-shard region, or report it cannot."""

        if self.fuse_cross_shard_gates and touched and cursor - index > 1:
            group = self.ir.instructions[index:cursor]
            matrix = _compose_gate_matrices(self.matrices[index:cursor])
            diagonal_count = sum(_is_diagonal_instruction(item.name) for item in group)
            non_diagonal_count = len(group) - diagonal_count
            if non_diagonal_count == 0:
                self.shard_state, scratch = _vectorized_local_diagonal_gate(
                    self.shard_state,
                    matrix,
                    instruction.wires,
                    plan=self.plan,
                    chunk_amplitudes=self.chunk_amplitudes,
                )
                self.local_count += len(group)
                self.local_diagonal_count += len(group)
            elif len(instruction.wires) == 1:
                self.shard_state, count, byte_count, scratch = (
                    _vectorized_pair_exchange_gate(
                        self.shard_state,
                        matrix,
                        instruction.wires[0],
                        plan=self.plan,
                        chunk_amplitudes=self.chunk_amplitudes,
                        process_group=self.process_group,
                        workspace=self.exchange_workspace,
                        pipeline=self.pipeline_pair_exchange,
                    )
                )
                self.local_count += diagonal_count
                self.local_diagonal_count += diagonal_count
                self.distributed_count += non_diagonal_count
                self.executed_distributed_segments += 1
                self.communication_count += count
                self.communication_bytes += byte_count
                self.fused_cross_shard_regions += 1
                self.fused_cross_shard_gate_count += len(group)
            else:
                self.shard_state, count, byte_count, scratch = (
                    _vectorized_subgroup_exchange_gate(
                        self.shard_state,
                        matrix,
                        instruction.wires,
                        plan=self.plan,
                        chunk_amplitudes=self.chunk_amplitudes,
                        process_group=self.process_group,
                        workspace=self.exchange_workspace,
                        pipeline=self.pipeline_pair_exchange,
                    )
                )
                self.local_count += diagonal_count
                self.local_diagonal_count += diagonal_count
                self.distributed_count += non_diagonal_count
                self.executed_distributed_segments += 1
                self.communication_count += count
                self.communication_bytes += byte_count
                self.fused_cross_shard_regions += 1
                self.fused_cross_shard_gate_count += len(group)
            self.peak_scratch = max(self.peak_scratch, scratch)
            return True
        return False

    def _diagonal(
        self,
        index: int,
        instruction: Instruction,
        matrix: torch.Tensor,
    ) -> bool:
        """Apply a diagonal gate locally, or report it cannot."""

        if _is_diagonal_instruction(instruction.name):
            self.shard_state, scratch = _vectorized_local_diagonal_gate(
                self.shard_state,
                matrix,
                instruction.wires,
                plan=self.plan,
                chunk_amplitudes=self.chunk_amplitudes,
                output=(
                    self.shard_state.amplitudes if self.local_compilation else None
                ),
            )
            self.peak_scratch = max(self.peak_scratch, scratch)
            self.local_count += 1
            self.local_diagonal_count += 1
            return True
        return False

    def _local(
        self,
        index: int,
        instruction: Instruction,
        matrix: torch.Tensor,
        touched: bool,
    ) -> bool:
        """Apply a rank-local gate, or report it cannot."""

        if not touched or self.world_size == 1:
            local_output = (
                self.shard_state.amplitudes if self.local_compilation else None
            )
            cx_decision = _triton_local_cx_decision(
                supported=bool(
                    instruction.name == "cx"
                    and self.shard_state.amplitudes.device.type == "cuda"
                    and self.shard_state.amplitudes.dtype == torch.complex64
                )
            )
            if instruction.name == "cx":
                self.kernel_dispatch_evidence.record(cx_decision)
            if cx_decision.accelerated:
                self.shard_state, scratch = _vectorized_local_cx_gate(
                    self.shard_state,
                    instruction.wires,
                    plan=self.plan,
                    output=local_output,
                )
            else:
                self.shard_state, scratch = _vectorized_local_gate(
                    self.shard_state,
                    matrix,
                    instruction.wires,
                    plan=self.plan,
                    chunk_amplitudes=self.chunk_amplitudes,
                    output=local_output,
                    kernel_dispatch_evidence=self.kernel_dispatch_evidence,
                )
            self.peak_scratch = max(self.peak_scratch, scratch)
            self.local_count += 1
            return True
        return False

    def _cross_shard(
        self,
        index: int,
        instruction: Instruction,
        matrix: torch.Tensor,
    ) -> None:
        """Apply one gate across shards, by the cheapest exchange it fits."""

        if (
            _cross_shard_cx_packing_enabled()
            and instruction.name == "cx"
            and sum(wire in self.plan.sharded_wires for wire in instruction.wires) == 1
        ):
            self.shard_state, count, byte_count, scratch = _vectorized_cross_shard_cx(
                self.shard_state,
                instruction.wires,
                plan=self.plan,
                chunk_amplitudes=self.chunk_amplitudes,
                process_group=self.process_group,
                workspace=self.exchange_workspace,
            )
        elif (
            len(instruction.wires) == 1
            and instruction.wires[0] in self.plan.sharded_wires
        ):
            self.shard_state, count, byte_count, scratch = (
                _vectorized_pair_exchange_gate(
                    self.shard_state,
                    matrix,
                    instruction.wires[0],
                    plan=self.plan,
                    chunk_amplitudes=self.chunk_amplitudes,
                    process_group=self.process_group,
                    workspace=self.exchange_workspace,
                    pipeline=self.pipeline_pair_exchange,
                )
            )
        else:
            self.shard_state, count, byte_count, scratch = (
                _vectorized_subgroup_exchange_gate(
                    self.shard_state,
                    matrix,
                    instruction.wires,
                    plan=self.plan,
                    chunk_amplitudes=self.chunk_amplitudes,
                    process_group=self.process_group,
                    workspace=self.exchange_workspace,
                    pipeline=self.pipeline_pair_exchange,
                )
            )
        self.distributed_count += 1
        self.executed_distributed_segments += 1
        self.communication_count += count
        self.communication_bytes += byte_count
        self.peak_scratch = max(self.peak_scratch, scratch)

    def _result(self) -> TorchDistributedStatevectorResult:
        """Build the execution result from the swept state."""

        if self.persistent_plan is not None:
            self.logical_to_physical = tuple(
                self.persistent_mapping[self.logical_to_physical[wire]]
                for wire in range(self.ir.n_wires)
            )
            self.wire_layout = "persistent"
        return TorchDistributedStatevectorResult(
            shard_state=self.shard_state,
            plan=self.plan,
            backend=str(self.backend),
            local_gate_count=self.local_count,
            distributed_gate_count=self.distributed_count,
            communication_count=self.communication_count,
            communication_bytes=self.communication_bytes,
            peak_scratch_bytes=self.peak_scratch,
            local_diagonal_gate_count=self.local_diagonal_count,
            executed_distributed_segment_count=self.executed_distributed_segments,
            fused_cross_shard_regions=self.fused_cross_shard_regions,
            fused_cross_shard_gate_count=self.fused_cross_shard_gate_count,
            cross_shard_fusion_enabled=bool(self.fuse_cross_shard_gates),
            exchange_workspace_allocation_count=(
                self.exchange_workspace.allocation_count
                if self.exchange_workspace is not None
                else 0
            ),
            exchange_workspace_reuse_count=(
                self.exchange_workspace.reuse_count
                if self.exchange_workspace is not None
                else 0
            ),
            exchange_workspace_reserved_bytes=(
                self.exchange_workspace.reserved_bytes
                if self.exchange_workspace is not None
                else 0
            ),
            exchange_pipeline_prefetch_count=(
                self.exchange_workspace.pipeline_prefetch_count
                if self.exchange_workspace is not None
                else 0
            ),
            exchange_pipelined_gate_count=(
                self.exchange_workspace.pipelined_gate_count
                if self.exchange_workspace is not None
                else 0
            ),
            exchange_pipeline_enabled=bool(self.pipeline_pair_exchange),
            exchange_subgroup_pipelined_gate_count=(
                self.exchange_workspace.subgroup_pipelined_gate_count
                if self.exchange_workspace is not None
                else 0
            ),
            intra_node_communication_count=(
                self.exchange_workspace.intra_node_message_count
                if self.exchange_workspace is not None
                else 0
            ),
            intra_node_communication_bytes=(
                self.exchange_workspace.intra_node_bytes
                if self.exchange_workspace is not None
                else 0
            ),
            inter_node_communication_count=(
                self.exchange_workspace.inter_node_message_count
                if self.exchange_workspace is not None
                else 0
            ),
            inter_node_communication_bytes=(
                self.exchange_workspace.inter_node_bytes
                if self.exchange_workspace is not None
                else 0
            ),
            wire_layout=self.wire_layout,
            logical_to_physical_wires=self.logical_to_physical,
            distributed_identity=build_distributed_identity(
                outer_backend=str(self.backend),
                logical_device=str(self.resolved_device),
                rank=self.rank,
                world_size=self.world_size,
                process_group_initialized=dist.is_initialized(),
            ),
            kernel_dispatch_evidence=self.kernel_dispatch_evidence,
            persistent_inter_node_ket_checkpoints=tuple(
                self.persistent_inter_node_ket_checkpoints
            ),
        )
