"""Rank-local gate application and the fused reversible VJP kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.distributed as dist

from ....core.ir import Instruction
from ....core.runtime_config import get_runtime_config, runtime_config
from ....simulation.statevector.operations import _instruction_matrix
from .forward import (
    StatevectorExchangeWorkspace,
    _is_diagonal_instruction,
    _triton_local_cx_enabled,
    _vectorized_cross_shard_cx,
    _vectorized_local_cx_gate,
    _vectorized_local_diagonal_gate,
    _vectorized_local_gate,
    _vectorized_pair_exchange_gate,
    _vectorized_subgroup_exchange_gate,
    _wait_for_exchange,
)
from .reverse_support import (
    BackwardExecutionEvidence,
    _fused_vjp_pipeline_enabled,
    _reverse_chunk_amplitudes,
    _reverse_cross_shard_cx_packing_enabled,
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
