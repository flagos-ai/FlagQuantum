"""Reverse sweep over rank-local statevector shards."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime
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
from ....simulation.statevector.operations import _instruction_matrix
from .checkpointing import StatevectorCheckpointPolicy
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
from .layout import distributed_swap_rank_local_bits, plan_persistent_statevector_layout
from .local_execution import initialize_statevector_shard, use_compact_global_indices
from .planning import plan_distributed_statevector
from .reverse_support import (
    BackwardExecutionEvidence,
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


class _MatrixJVP(Protocol):
    """PyTorch JVP boundary for a matrix controlled by one scalar tensor."""

    def __call__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        inputs: tuple[torch.Tensor],
        v: tuple[torch.Tensor],
        *,
        create_graph: bool,
        strict: bool,
    ) -> object: ...


def _compact_reverse_global_indices(plan: Any, rank: int) -> bool:
    """Use the shared shard-index representation policy for reverse states."""
    return bool(use_compact_global_indices(plan, rank))


def _local_expectation_z_sum_adjoint(
    shard_state: Any, *, plan: Any, n_wires: int, wires: tuple[int, ...]
) -> torch.Tensor:
    """Construct one adjoint seed for a sum of single-wire Z terms."""

    adjoint = torch.zeros_like(shard_state.amplitudes)
    local_count = shard_state.shard.local_amplitudes
    with torch.no_grad():
        for start in range(0, local_count, _reverse_chunk_amplitudes()):
            end = min(local_count, start + _reverse_chunk_amplitudes())
            indices = _storage_global_indices(shard_state, start, end, plan=plan)
            for wire in wires:
                adjoint[:, start:end].add_(
                    _z_expectation_adjoint_chunk(
                        shard_state.amplitudes[:, start:end],
                        indices,
                        n_wires=n_wires,
                        wire=wire,
                    )
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
    if workspace is not None and pipelined:
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
        if workspace is not None and pipelined and chunk_index + 1 < len(chunks):
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


class _ReversibleAdjointSweep:
    """Rematerialize rank-local kets with U† and propagate gate adjoints."""

    def __init__(
        self,
        ir: CircuitIR,
        slots: tuple[tuple[int, str, int], ...],
        saved_parameters: tuple[torch.Tensor, ...],
        *,
        observable_wires: tuple[int, ...],
        device: torch.device,
        policy: StatevectorCheckpointPolicy,
        process_group: Any | None,
        evidence: BackwardExecutionEvidence,
        saved_final_state: Any | None,
        saved_inter_node_ket_checkpoints: tuple[tuple[int, int, torch.Tensor], ...],
    ) -> None:
        self.ir = ir
        self.slots = slots
        self.saved_parameters = saved_parameters
        self.observable_wires = observable_wires
        self.device = device
        self.policy = policy
        self.process_group = process_group
        self.evidence = evidence
        self.saved_final_state = saved_final_state
        self.saved_inter_node_ket_checkpoints = saved_inter_node_ket_checkpoints

    def run(self) -> tuple[torch.Tensor, ...]:
        """Run the reverse sweep and return one gradient per parameter."""

        self._prepare_layout()
        self._prepare_checkpoints()
        self._prepare_adjoint()
        self._prepare_gradients()
        self.skipped_cx_indices: set[int] = set()
        self.cx_segment_scratch: tuple[torch.Tensor, torch.Tensor] | None = None
        for index in range(len(self.bound.instructions) - 1, -1, -1):
            if index in self.skipped_cx_indices:
                continue
            execution_instruction = replace(
                self.bound.instructions[index],
                wires=tuple(
                    self.persistent_mapping[int(wire)]
                    for wire in self.bound.instructions[index].wires
                ),
            )
            active = sorted(self.slots_by_instruction.get(index, ()))
            if self._apply_accelerated_reversible(index, execution_instruction, active):
                continue
            if self._apply_local_cx_segment(index, execution_instruction):
                continue
            self._apply_general_step(index, execution_instruction, active)
        self._finish()
        return tuple(self.accumulated)

    def _prepare_layout(self) -> None:
        """Resolve the distributed plan and the persistent-layout state it needs."""

        self.world_size = (
            dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        )
        self.rank = dist.get_rank(self.process_group) if dist.is_initialized() else 0
        self.dtype = (
            torch.complex128
            if any(p.dtype == torch.float64 for p in self.saved_parameters)
            else torch.complex64
        )
        self.base_parameters = tuple(
            value.detach().to(self.device) for value in self.saved_parameters
        )
        self.bound = _bind_parameters(self.ir, self.slots, self.base_parameters)
        self.plan = plan_distributed_statevector(
            self.bound,
            world_size=self.world_size,
            local_world_size=self.world_size,
            bsz=1,
            complex_bytes=torch.empty((), dtype=self.dtype).element_size(),
        )
        self.exchange_workspace = (
            StatevectorExchangeWorkspace()
            if _reverse_exchange_workspace_enabled()
            else None
        )
        persistent_layout = (
            _persistent_wire_layout_enabled()
            and self.world_size > 1
            and self.policy.strategy == "reversible_adjoint"
        )
        persistent_plan = (
            plan_persistent_statevector_layout(self.bound, world_size=self.world_size)
            if persistent_layout
            else None
        )
        self.evidence.persistent_layout_enabled = persistent_plan is not None
        self.persistent_mapping = list(
            persistent_plan.final_logical_to_physical
            if persistent_plan is not None
            else range(self.ir.n_wires)
        )
        self.swaps_before: dict[int, list[Any]] = {}
        if persistent_plan is not None:
            for swap in persistent_plan.swaps:
                self.swaps_before.setdefault(swap.before_instruction, []).append(swap)
        self.inter_node_ket_checkpoints = {
            (index, swap_index): amplitudes
            for index, swap_index, amplitudes in self.saved_inter_node_ket_checkpoints
        }
        local_wire_capacity = self.ir.n_wires - self.plan.rank_address_bits
        self.inplace_local = bool(
            (persistent_plan is not None or self.world_size == 1)
            and _persistent_inplace_local_enabled()
            and all(
                len(instruction.wires) <= local_wire_capacity
                for instruction in self.bound.instructions
            )
        )
        self.evidence.persistent_inplace_local_enabled = self.inplace_local
        self.layout_send_buffer = self.layout_receive_buffer = None
        if persistent_plan is not None and persistent_plan.swaps:
            local_amplitudes = self.plan.shards[self.rank].local_amplitudes
            half_shape = (self.plan.bsz, local_amplitudes // 2)
            self.layout_send_buffer = torch.empty(
                half_shape, dtype=self.dtype, device=self.device
            )
            self.layout_receive_buffer = torch.empty_like(self.layout_send_buffer)
            self.evidence.persistent_layout_buffer_allocation_count = 2
            self.evidence.persistent_layout_buffer_reserved_bytes = (
                2
                * self.layout_send_buffer.numel()
                * self.layout_send_buffer.element_size()
            )

    def _prepare_checkpoints(self) -> None:
        """Initialize the shard and record every interval checkpoint."""

        self.initial = None
        self.checkpoints: dict[int, Any] = {}
        if self.saved_final_state is None:
            self.initial = initialize_statevector_shard(
                self.plan,
                rank=self.rank,
                device=self.device,
                dtype=self.dtype,
                compact_global_indices=_compact_reverse_global_indices(
                    self.plan, self.rank
                ),
            )
            self.checkpoints[0] = self.initial
        if self.policy.strategy == "interval":
            assert self.initial is not None
            state = self.initial
            with torch.no_grad():
                for index in range(len(self.bound.instructions)):
                    state = _apply_one_gate(
                        state,
                        instruction=self.bound.instructions[index],
                        plan=self.plan,
                        dtype=self.dtype,
                        process_group=self.process_group,
                        evidence=self.evidence,
                        workspace=self.exchange_workspace,
                    )
                    if (index + 1) % self.policy.interval == 0:
                        self.checkpoints[index + 1] = replace(
                            state, amplitudes=state.amplitudes.detach().clone()
                        )

    def _state_before(self, stop: int) -> Any:
        """Replay forward from the nearest checkpoint at or before `stop`."""

        start = max(index for index in self.checkpoints if index <= stop)
        checkpoint = self.checkpoints[start]
        # Forward kernels are out-of-place, so a checkpoint can safely be the
        # replay input.  Cloning here retained another complete rank-local
        # state for every adjoint gate (8 GiB at 31q/world-size 2).
        state = replace(checkpoint, amplitudes=checkpoint.amplitudes.detach())
        with torch.no_grad():
            for index in range(start, stop):
                state = _apply_one_gate(
                    state,
                    instruction=self.bound.instructions[index],
                    plan=self.plan,
                    dtype=self.dtype,
                    process_group=self.process_group,
                    evidence=self.evidence,
                    workspace=self.exchange_workspace,
                )
        return state

    def _prepare_adjoint(self) -> None:
        """Take the final state, seed the adjoint, and allocate the work buffers."""

        final_state = (
            self._state_before(len(self.bound.instructions))
            if self.saved_final_state is None
            else replace(
                self.saved_final_state,
                amplitudes=self.saved_final_state.amplitudes.detach(),
            )
        )
        self.evidence.saved_forward_state_reused = self.saved_final_state is not None
        final_observable_wires = tuple(
            self.persistent_mapping[wire] for wire in self.observable_wires
        )
        self.adjoint = _local_expectation_z_sum_adjoint(
            final_state,
            plan=self.plan,
            n_wires=self.ir.n_wires,
            wires=final_observable_wires,
        )
        self.reversible_state = (
            final_state if self.policy.strategy == "reversible_adjoint" else None
        )
        if self.reversible_state is None:
            del final_state
            self.reversible_scratch = None
            self.adjoint_scratch = None
        else:
            # A reversible sweep reconstructs every preceding ket from the final
            # ket with U†, so retaining the zero-state replay checkpoint wastes one
            # complete shard (16 GiB for a 31q complex64 single-GPU run).
            self.checkpoints.clear()
            if self.initial is not None:
                del self.initial
            del final_state
            # Allocate the only two full-shard work buffers up front.  Every
            # subsequent inverse ket/adjoint gate alternates between these fixed
            # allocations instead of asking the CUDA allocator for another 16 GiB.
            self.reversible_scratch = (
                None
                if self.inplace_local
                else torch.empty_like(self.reversible_state.amplitudes)
            )
            self.adjoint_scratch = (
                None if self.inplace_local else torch.empty_like(self.adjoint)
            )
            if os.getenv("FQ_STATEVECTOR_DEBUG_MEMORY", "0") == "1":
                platform = get_platform_runtime(self.device.type)
                platform.synchronize(self.device)
                memory = platform.memory_snapshot(self.device)
                print(
                    "reversible_adjoint_initialized",
                    {
                        "allocated": memory.allocated_bytes,
                        "reserved": memory.reserved_bytes,
                        "shard_bytes": self.reversible_state.amplitudes.numel()
                        * self.reversible_state.amplitudes.element_size(),
                    },
                    flush=True,
                )

    def _prepare_gradients(self) -> None:
        """Set up per-parameter accumulation and the asynchronous reducer."""

        self.accumulated = [
            torch.zeros_like(parameter) for parameter in self.base_parameters
        ]
        self.slots_by_instruction: dict[int, set[int]] = {}
        for instruction_index, _, parameter_index in self.slots:
            self.slots_by_instruction.setdefault(instruction_index, set()).add(
                parameter_index
            )
        self.remaining_parameter_instructions = [0] * len(self.base_parameters)
        for parameter_indices in self.slots_by_instruction.values():
            for parameter_index in parameter_indices:
                self.remaining_parameter_instructions[parameter_index] += 1
        self.gradient_reducer = (
            AsyncGradientReducer(
                self.accumulated,
                process_group=self.process_group,
                evidence=self.evidence,
                max_parameters=(None if _gradient_bucketing_enabled() else 1),
                max_bytes=(None if _gradient_bucketing_enabled() else 1),
                async_op=_gradient_reduction_overlap_enabled(),
            )
            if self.world_size > 1
            else None
        )

    def _accumulate_parameter_gradient(
        self,
        parameter_index: int,
        gradient: torch.Tensor,
        index: int,
    ) -> None:
        """Add one parameter's local gradient and release it when it is complete."""

        self.accumulated[parameter_index] += gradient.detach()
        self.evidence.reduction_counts[parameter_index] += 1
        self.remaining_parameter_instructions[parameter_index] -= 1
        if (
            self.gradient_reducer is not None
            and self.remaining_parameter_instructions[parameter_index] == 0
        ):
            self.gradient_reducer.mark_ready(
                parameter_index,
                overlap_opportunity=index > 0,
            )

    def _apply_accelerated_reversible(
        self,
        index: int,
        execution_instruction: Instruction,
        active: list[int],
    ) -> bool:
        """Reconstruct one ket with the fused reversible VJP, or report it cannot."""

        reversible_vjp_decision = _triton_vjp_adjoint_decision(
            supported=bool(
                self.inplace_local
                and self.reversible_state is not None
                and len(active) == 1
                and len(execution_instruction.wires) == 1
                and execution_instruction.wires[0] not in self.plan.sharded_wires
                and self.reversible_state.amplitudes.dtype == torch.complex64
                and self.reversible_state.amplitudes.device.type == "cuda"
            )
        )
        if self.reversible_state is not None and reversible_vjp_decision.accelerated:
            precision = get_runtime_config().with_overrides(
                complex_dtype=str(self.dtype).removeprefix("torch.")
            )
            with runtime_config(precision):
                accelerated_matrix = _instruction_matrix(
                    execution_instruction, device=self.device, dtype=self.dtype
                )
            derivative_matrix = _analytic_rotation_derivative(
                execution_instruction, accelerated_matrix
            )
            if derivative_matrix is not None:
                self.evidence.kernel_dispatch_evidence.record(reversible_vjp_decision)
                from ....simulation.triton_kernels.statevector_adjoint import (
                    fused_complex64_local_1q_reversible_vjp,
                )

                bit_position = (
                    self.plan.n_wires
                    - execution_instruction.wires[0]
                    - 1
                    - len(self.plan.sharded_wires)
                )
                gradient = fused_complex64_local_1q_reversible_vjp(
                    self.reversible_state.amplitudes,
                    self.adjoint,
                    accelerated_matrix,
                    derivative_matrix,
                    bit_position=bit_position,
                ).to(dtype=self.base_parameters[active[0]].dtype)
                self._accumulate_parameter_gradient(active[0], gradient, index)
                self.evidence.analytic_rotation_derivative_count += 1
                self.evidence.fused_parameter_adjoint_count += 1
                for swap_index, swap in reversed(
                    tuple(enumerate(self.swaps_before.get(index, ())))
                ):
                    checkpoint = self.inter_node_ket_checkpoints.pop(
                        (index, swap_index), None
                    )
                    if checkpoint is None:
                        _, count, byte_count = distributed_swap_rank_local_bits(
                            self.reversible_state.amplitudes,
                            rank=self.rank,
                            n_wires=self.ir.n_wires,
                            rank_bits=self.plan.rank_address_bits,
                            local_physical_wire=swap.local_physical_wire,
                            sharded_physical_wire=swap.sharded_physical_wire,
                            process_group=self.process_group,
                            output=self.reversible_state.amplitudes,
                            send_buffer=self.layout_send_buffer,
                            receive_buffer=self.layout_receive_buffer,
                            tag=(
                                index * max(1, self.plan.rank_address_bits) + swap_index
                            )
                            * 2,
                        )
                    else:
                        self.reversible_state = replace(
                            self.reversible_state, amplitudes=checkpoint
                        )
                        count = byte_count = 0
                    _, adjoint_count, adjoint_bytes = distributed_swap_rank_local_bits(
                        self.adjoint,
                        rank=self.rank,
                        n_wires=self.ir.n_wires,
                        rank_bits=self.plan.rank_address_bits,
                        local_physical_wire=swap.local_physical_wire,
                        sharded_physical_wire=swap.sharded_physical_wire,
                        process_group=self.process_group,
                        output=self.adjoint,
                        send_buffer=self.layout_send_buffer,
                        receive_buffer=self.layout_receive_buffer,
                        tag=(index * max(1, self.plan.rank_address_bits) + swap_index)
                        * 2
                        + 1,
                    )
                    self.evidence.communication_count += count + adjoint_count
                    self.evidence.communication_bytes += byte_count + adjoint_bytes
                    self.evidence.persistent_layout_swap_count += 1
                    self.evidence.persistent_layout_swap_bytes += (
                        byte_count + adjoint_bytes
                    )
                    (
                        self.persistent_mapping[swap.local_logical_wire],
                        self.persistent_mapping[swap.sharded_logical_wire],
                    ) = (
                        self.persistent_mapping[swap.sharded_logical_wire],
                        self.persistent_mapping[swap.local_logical_wire],
                    )
                return True
        return False

    def _apply_local_cx_segment(
        self,
        index: int,
        execution_instruction: Instruction,
    ) -> bool:
        """Fuse a contiguous run of local CX gates, or report it cannot."""

        if (
            self.inplace_local
            and self.reversible_state is not None
            and triton_available()
            and _triton_local_cx_segment_enabled(self.bound)
            and execution_instruction.name == "cx"
            and not self.swaps_before.get(index)
        ):
            segment_start = index
            while segment_start > 0:
                candidate_index = segment_start - 1
                candidate = self.bound.instructions[candidate_index]
                if candidate.name != "cx" or self.swaps_before.get(candidate_index):
                    break
                segment_start = candidate_index
            if segment_start < index:
                from ....simulation.triton_kernels.statevector_gates import (
                    apply_complex64_local_cx_segment,
                )

                segment_wires = tuple(
                    tuple(self.persistent_mapping[int(wire)] for wire in item.wires)
                    for item in self.bound.instructions[segment_start : index + 1]
                )
                if not any(
                    wire in self.plan.sharded_wires
                    for wires in segment_wires
                    for wire in wires
                ):
                    rank_bits = len(self.plan.sharded_wires)
                    controls = tuple(
                        self.plan.n_wires - wires[0] - 1 - rank_bits
                        for wires in reversed(segment_wires)
                    )
                    targets = tuple(
                        self.plan.n_wires - wires[1] - 1 - rank_bits
                        for wires in reversed(segment_wires)
                    )
                    if self.cx_segment_scratch is None:
                        self.cx_segment_scratch = (
                            torch.empty_like(self.reversible_state.amplitudes),
                            torch.empty_like(self.adjoint),
                        )
                    cx_segment_ket_scratch, cx_segment_adjoint_scratch = (
                        self.cx_segment_scratch
                    )
                    previous_ket = self.reversible_state.amplitudes
                    apply_complex64_local_cx_segment(
                        previous_ket,
                        control_bit_positions=controls,
                        target_bit_positions=targets,
                        output=cx_segment_ket_scratch,
                    )
                    self.reversible_state = replace(
                        self.reversible_state, amplitudes=cx_segment_ket_scratch
                    )
                    previous_adjoint = self.adjoint
                    apply_complex64_local_cx_segment(
                        previous_adjoint,
                        control_bit_positions=controls,
                        target_bit_positions=targets,
                        output=cx_segment_adjoint_scratch,
                    )
                    self.adjoint = cx_segment_adjoint_scratch
                    self.cx_segment_scratch = (previous_ket, previous_adjoint)
                    self.evidence.peak_scratch_bytes = max(
                        self.evidence.peak_scratch_bytes,
                        2
                        * self.reversible_state.amplitudes.numel()
                        * self.reversible_state.amplitudes.element_size(),
                    )
                    self.skipped_cx_indices.update(range(segment_start, index))
                    return True
        return False

    def _apply_general_step(
        self,
        index: int,
        execution_instruction: Instruction,
        active: list[int],
    ) -> None:
        """Rematerialize, differentiate, propagate, and undo this instruction's swaps."""

        before, original_matrix = self._rematerialize_before(
            index, execution_instruction
        )
        fused_adjoint, original_matrix = self._parameter_derivatives(
            index, execution_instruction, active, before, original_matrix
        )
        self._propagate_adjoint(
            index,
            execution_instruction,
            before,
            fused_adjoint,
            original_matrix,
        )
        self._replay_swaps(index)

        if self.reversible_state is None:
            del before

    def _rematerialize_before(
        self,
        index: int,
        execution_instruction: Instruction,
    ) -> tuple[Any, torch.Tensor | None]:
        """Rebuild the ket this instruction consumed, and the matrix it applied."""

        original_matrix: torch.Tensor | None = None
        if self.reversible_state is None:
            before = self._state_before(index)
        else:
            precision = get_runtime_config().with_overrides(
                complex_dtype=str(self.dtype).removeprefix("torch.")
            )
            with runtime_config(precision):
                inverse_matrix = _instruction_matrix(
                    execution_instruction, device=self.device, dtype=self.dtype
                ).mH
            original_matrix = inverse_matrix.mH
            previous_reversible_state = self.reversible_state
            previous_reversible_amplitudes = previous_reversible_state.amplitudes
            inverse_output = (
                previous_reversible_amplitudes
                if self.inplace_local
                else self.reversible_scratch
            )
            before = _apply_matrix_gate(
                replace(
                    previous_reversible_state,
                    amplitudes=previous_reversible_state.amplitudes.detach(),
                ),
                matrix=inverse_matrix,
                instruction=execution_instruction,
                plan=self.plan,
                process_group=self.process_group,
                evidence=self.evidence,
                workspace=self.exchange_workspace,
                output=inverse_output,
            )
            self.reversible_state = before
            if not self.inplace_local:
                self.reversible_scratch = previous_reversible_amplitudes
            del previous_reversible_state, inverse_matrix
        return before, original_matrix

    def _parameter_derivatives(
        self,
        index: int,
        execution_instruction: Instruction,
        active: list[int],
        before: Any,
        original_matrix: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Accumulate every active parameter's gradient for this instruction."""

        fused_adjoint: torch.Tensor | None = None
        derivatives: list[torch.Tensor] = []
        if active:
            instruction = execution_instruction
            if original_matrix is None:
                precision = get_runtime_config().with_overrides(
                    complex_dtype=str(self.dtype).removeprefix("torch.")
                )
                with runtime_config(precision):
                    original_matrix = _instruction_matrix(
                        instruction, device=self.device, dtype=self.dtype
                    )
            for parameter_index in active:
                parameter = self.base_parameters[parameter_index]
                if parameter.numel() != 1:
                    raise ValueError(
                        "sharded statevector reverse currently expects scalar "
                        "gate parameters"
                    )

                def matrix_for_parameter(
                    value: torch.Tensor,
                    parameter_index: int = parameter_index,
                    instruction_position: int = index,
                ) -> torch.Tensor:
                    values = list(self.base_parameters)
                    values[parameter_index] = value
                    current_bound = _bind_parameters(self.ir, self.slots, values)
                    precision = get_runtime_config().with_overrides(
                        complex_dtype=str(self.dtype).removeprefix("torch.")
                    )
                    with runtime_config(precision):
                        return _instruction_matrix(
                            current_bound.instructions[instruction_position],
                            device=self.device,
                            dtype=self.dtype,
                        )

                derivative_matrix = _analytic_rotation_derivative(
                    instruction,
                    original_matrix,
                )
                if derivative_matrix is not None:
                    self.evidence.analytic_rotation_derivative_count += 1
                if derivative_matrix is None:
                    # Generic gates retain the tiny-matrix JVP fallback.
                    matrix_jvp: _MatrixJVP = torch.autograd.functional.jvp
                    jvp_result = matrix_jvp(
                        matrix_for_parameter,
                        (parameter.detach().requires_grad_(True),),
                        (torch.ones_like(parameter),),
                        create_graph=False,
                        strict=False,
                    )
                    if (
                        not isinstance(jvp_result, tuple)
                        or len(jvp_result) != 2
                        or not isinstance(jvp_result[1], torch.Tensor)
                    ):
                        raise TypeError("Matrix JVP must return a tensor derivative")
                    derivative_matrix = jvp_result[1]
                vjp_decision = _triton_vjp_adjoint_decision(
                    supported=bool(
                        len(active) == 1
                        and len(instruction.wires) == 1
                        and before.amplitudes.dtype == torch.complex64
                        and before.amplitudes.device.type == "cuda"
                    )
                )
                self.evidence.kernel_dispatch_evidence.record(vjp_decision)
                if vjp_decision.accelerated:
                    from ....simulation.triton_kernels.statevector_adjoint import (
                        fused_complex64_local_1q_vjp_adjoint,
                    )

                    wire = instruction.wires[0]
                    if wire in self.plan.sharded_wires and self.plan.world_size > 1:
                        (
                            fused_adjoint,
                            local_derivative,
                            communication_count,
                            communication_bytes,
                            scratch_bytes,
                        ) = _fused_sharded_1q_vjp_adjoint(
                            before,
                            self.adjoint,
                            original_matrix,
                            derivative_matrix,
                            wire=wire,
                            plan=self.plan,
                            process_group=self.process_group,
                            workspace=self.exchange_workspace,
                            output=(
                                self.adjoint
                                if self.inplace_local
                                else self.adjoint_scratch
                            ),
                        )
                        self.evidence.communication_count += communication_count
                        self.evidence.communication_bytes += communication_bytes
                        self.evidence.peak_scratch_bytes = max(
                            self.evidence.peak_scratch_bytes, scratch_bytes
                        )
                    else:
                        bit_position = (
                            self.plan.n_wires - wire - 1 - len(self.plan.sharded_wires)
                        )
                        fused_adjoint, local_derivative = (
                            fused_complex64_local_1q_vjp_adjoint(
                                before.amplitudes,
                                self.adjoint,
                                original_matrix,
                                derivative_matrix,
                                bit_position=bit_position,
                                output=(
                                    self.adjoint
                                    if self.inplace_local
                                    else self.adjoint_scratch
                                ),
                            )
                        )
                        self.evidence.peak_scratch_bytes = max(
                            self.evidence.peak_scratch_bytes,
                            fused_adjoint.numel() * fused_adjoint.element_size(),
                        )
                    local_derivative = local_derivative.to(dtype=parameter.dtype)
                    derivatives.append(local_derivative)
                    self.evidence.fused_parameter_adjoint_count += 1
                    del derivative_matrix
                    continue
                with torch.no_grad():
                    derivative_state = _apply_matrix_gate(
                        replace(before, amplitudes=before.amplitudes.detach()),
                        matrix=derivative_matrix,
                        instruction=execution_instruction,
                        plan=self.plan,
                        process_group=self.process_group,
                        evidence=self.evidence,
                        workspace=self.exchange_workspace,
                    )
                    local_derivative = torch.zeros_like(parameter)
                    local_count = derivative_state.shard.local_amplitudes
                    for start in range(0, local_count, _reverse_chunk_amplitudes()):
                        end = min(local_count, start + _reverse_chunk_amplitudes())
                        local_derivative += _real_conjugate_inner_sum(
                            self.adjoint[:, start:end],
                            derivative_state.amplitudes[:, start:end],
                        ).to(dtype=parameter.dtype)
                derivatives.append(local_derivative)
                del derivative_state, derivative_matrix
        for parameter_index, gradient in zip(active, derivatives, strict=True):
            self._accumulate_parameter_gradient(parameter_index, gradient, index)
        return fused_adjoint, original_matrix

    def _propagate_adjoint(
        self,
        index: int,
        execution_instruction: Instruction,
        before: Any,
        fused_adjoint: torch.Tensor | None,
        original_matrix: torch.Tensor | None,
    ) -> None:
        """Push the adjoint through this instruction's inverse."""

        if fused_adjoint is not None:
            previous_adjoint = self.adjoint
            self.adjoint = fused_adjoint
            if not self.inplace_local and self.adjoint_scratch is not None:
                self.adjoint_scratch = previous_adjoint
        else:
            if original_matrix is None:
                precision = get_runtime_config().with_overrides(
                    complex_dtype=str(self.dtype).removeprefix("torch.")
                )
                with runtime_config(precision):
                    original_matrix = _instruction_matrix(
                        self.bound.instructions[index],
                        device=self.device,
                        dtype=self.dtype,
                    )
            adjoint_state = replace(before, amplitudes=self.adjoint.detach())
            previous_adjoint = self.adjoint
            self.adjoint = _apply_matrix_gate(
                adjoint_state,
                matrix=original_matrix.mH,
                instruction=execution_instruction,
                plan=self.plan,
                process_group=self.process_group,
                evidence=self.evidence,
                workspace=self.exchange_workspace,
                output=self.adjoint if self.inplace_local else self.adjoint_scratch,
            ).amplitudes
            if not self.inplace_local and self.adjoint_scratch is not None:
                self.adjoint_scratch = previous_adjoint

    def _replay_swaps(self, index: int) -> None:
        """Undo the layout swaps this instruction's adjoint has to cross."""

        for swap_index, swap in reversed(
            tuple(enumerate(self.swaps_before.get(index, ())))
        ):
            assert self.reversible_state is not None
            previous_reversible = self.reversible_state.amplitudes
            checkpoint = self.inter_node_ket_checkpoints.pop((index, swap_index), None)
            if checkpoint is None:
                restored_reversible, count, byte_count = (
                    distributed_swap_rank_local_bits(
                        previous_reversible,
                        rank=self.rank,
                        n_wires=self.ir.n_wires,
                        rank_bits=self.plan.rank_address_bits,
                        local_physical_wire=swap.local_physical_wire,
                        sharded_physical_wire=swap.sharded_physical_wire,
                        process_group=self.process_group,
                        output=(
                            previous_reversible
                            if self.inplace_local
                            else self.reversible_scratch
                        ),
                        send_buffer=self.layout_send_buffer,
                        receive_buffer=self.layout_receive_buffer,
                        tag=(index * max(1, self.plan.rank_address_bits) + swap_index)
                        * 2,
                    )
                )
            else:
                restored_reversible = checkpoint
                count = byte_count = 0
            self.reversible_state = replace(
                self.reversible_state, amplitudes=restored_reversible
            )
            if not self.inplace_local:
                self.reversible_scratch = previous_reversible
            previous_adjoint = self.adjoint
            self.adjoint, adjoint_count, adjoint_bytes = (
                distributed_swap_rank_local_bits(
                    previous_adjoint,
                    rank=self.rank,
                    n_wires=self.ir.n_wires,
                    rank_bits=self.plan.rank_address_bits,
                    local_physical_wire=swap.local_physical_wire,
                    sharded_physical_wire=swap.sharded_physical_wire,
                    process_group=self.process_group,
                    output=(
                        previous_adjoint if self.inplace_local else self.adjoint_scratch
                    ),
                    send_buffer=self.layout_send_buffer,
                    receive_buffer=self.layout_receive_buffer,
                    tag=(index * max(1, self.plan.rank_address_bits) + swap_index) * 2
                    + 1,
                )
            )
            if not self.inplace_local:
                self.adjoint_scratch = previous_adjoint
            self.evidence.communication_count += count + adjoint_count
            self.evidence.communication_bytes += byte_count + adjoint_bytes
            self.evidence.persistent_layout_swap_count += 1
            self.evidence.persistent_layout_swap_bytes += byte_count + adjoint_bytes
            (
                self.persistent_mapping[swap.local_logical_wire],
                self.persistent_mapping[swap.sharded_logical_wire],
            ) = (
                self.persistent_mapping[swap.sharded_logical_wire],
                self.persistent_mapping[swap.local_logical_wire],
            )

    def _finish(self) -> None:
        """Flush the gradient reducer and release the exchange workspace."""

        if self.gradient_reducer is not None:
            self.gradient_reducer.finish()
        if self.exchange_workspace is not None:
            self.evidence.exchange_workspace_allocation_count = (
                self.exchange_workspace.allocation_count
            )
            self.evidence.exchange_workspace_reuse_count = (
                self.exchange_workspace.reuse_count
            )
            self.evidence.exchange_workspace_reserved_bytes = (
                self.exchange_workspace.reserved_bytes
            )
            self.evidence.exchange_pipeline_prefetch_count = (
                self.exchange_workspace.pipeline_prefetch_count
            )
            self.evidence.intra_node_communication_count = (
                self.exchange_workspace.intra_node_message_count
            )
            self.evidence.intra_node_communication_bytes = (
                self.exchange_workspace.intra_node_bytes
            )
            self.evidence.inter_node_communication_count = (
                self.exchange_workspace.inter_node_message_count
            )
            self.evidence.inter_node_communication_bytes = (
                self.exchange_workspace.inter_node_bytes
            )
