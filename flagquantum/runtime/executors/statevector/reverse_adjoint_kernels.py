"""Rank-local gate application and the fused reversible VJP kernels."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

import torch
import torch.distributed as dist

from ....core.ir import Instruction
from ....core.runtime_config import get_runtime_config, runtime_config
from ....simulation.native_cpu import fused_rotation_adjoint_
from ....simulation.statevector.operations import (
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _instruction_matrix,
)
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


def _cpu_direct_adjoint_gate_enabled() -> bool:
    """Use the native dense CPU layouts instead of rebuilding gather indices."""

    return os.getenv("FQ_STATEVECTOR_ADJOINT_CPU_DIRECT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _apply_single_process_cpu_gate(
    shard_state: Any,
    *,
    matrix: torch.Tensor,
    instruction: Instruction,
    plan: Any,
    output: torch.Tensor | None,
) -> tuple[Any, int]:
    """Apply a local CPU gate without materializing full-state gather indices."""

    source = shard_state.amplitudes
    if instruction.name in {"cx", "swap", "x"} and instruction.matrix is None:
        updated = _apply_fixed_permutation(
            source, instruction.name, instruction.wires, plan.n_wires
        )
    elif _is_diagonal_instruction(instruction.name):
        updated = _apply_diagonal_matrix(
            source, matrix, instruction.wires, plan.n_wires
        )
    else:
        updated = _apply_matrix(source, matrix, instruction.wires, plan.n_wires)
    if output is not None:
        output.copy_(updated)
        updated = output
    return (
        replace(shard_state, amplitudes=updated),
        updated.numel() * updated.element_size(),
    )


def _pair_components(
    state: torch.Tensor, *, wire: int, n_wires: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the zero/one views for one wire of a contiguous state."""

    stride = 1 << (int(n_wires) - int(wire) - 1)
    paired = state.reshape(state.shape[0], -1, 2, stride)
    return paired[:, :, 0, :], paired[:, :, 1, :]


def _complex_cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return Im(conj(left) * right) without materializing a complex product."""

    return left.real * right.imag - left.imag * right.real


def _single_process_cpu_pauli_rotation_gradient(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor | None:
    """Evaluate a Pauli-rotation VJP directly from the post-gate ket and bra."""

    name = instruction.name
    wires = instruction.wires
    if name in {"rx", "ry", "rz"} and len(wires) == 1:
        ket_zero, ket_one = _pair_components(ket, wire=wires[0], n_wires=n_wires)
        adjoint_zero, adjoint_one = _pair_components(
            adjoint, wire=wires[0], n_wires=n_wires
        )
        if name == "rx":
            return 0.5 * (
                _complex_cross(adjoint_zero, ket_one).sum()
                + _complex_cross(adjoint_one, ket_zero).sum()
            )
        if name == "ry":
            zero = (
                adjoint_zero.real * ket_one.real + adjoint_zero.imag * ket_one.imag
            ).sum()
            one = (
                adjoint_one.real * ket_zero.real + adjoint_one.imag * ket_zero.imag
            ).sum()
            return 0.5 * (one - zero)
        return 0.5 * (
            _complex_cross(adjoint_zero, ket_zero).sum()
            - _complex_cross(adjoint_one, ket_one).sum()
        )
    if name == "rzz" and len(wires) == 2:
        cross = _complex_cross(adjoint, ket)
        ordered = tuple(int(wire) for wire in wires)
        rest = tuple(wire for wire in range(n_wires) if wire not in ordered)
        perm = (0,) + tuple(wire + 1 for wire in ordered + rest)
        basis = (
            cross.reshape((cross.shape[0],) + (2,) * n_wires)
            .permute(perm)
            .reshape(cross.shape[0], 4, -1)
        )
        return 0.5 * (
            basis[:, 0].sum()
            - basis[:, 1].sum()
            - basis[:, 2].sum()
            + basis[:, 3].sum()
        )
    return None


def _fused_single_process_cpu_reversible_vjp(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor | None:
    """Reverse ket and bra while avoiding a derivative-state materialization."""

    if len(instruction.wires) == 1 or (
        instruction.name == "rzz" and len(instruction.wires) == 2
    ):
        native_gradient = fused_rotation_adjoint_(
            ket,
            adjoint,
            matrix,
            name=instruction.name,
            wire=instruction.wires[0],
            second_wire=(instruction.wires[1] if len(instruction.wires) == 2 else None),
            n_wires=n_wires,
        )
        if native_gradient is not None:
            return native_gradient

    gradient = _single_process_cpu_pauli_rotation_gradient(
        ket, adjoint, instruction, n_wires=n_wires
    )
    if gradient is None:
        return None
    inverse = matrix.mH
    apply = (
        _apply_diagonal_matrix
        if _is_diagonal_instruction(instruction.name)
        else _apply_matrix
    )
    previous_ket = apply(ket, inverse, instruction.wires, n_wires)
    previous_adjoint = apply(adjoint, inverse, instruction.wires, n_wires)
    ket.copy_(previous_ket)
    adjoint.copy_(previous_adjoint)
    return gradient


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
    if (
        plan.world_size == 1
        and shard_state.amplitudes.device.type == "cpu"
        and _cpu_direct_adjoint_gate_enabled()
    ):
        state, scratch = _apply_single_process_cpu_gate(
            shard_state,
            matrix=matrix,
            instruction=instruction,
            plan=plan,
            output=output,
        )
        if evidence is not None:
            evidence.peak_scratch_bytes = max(evidence.peak_scratch_bytes, scratch)
        return state
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
        if instruction.name == "cx" and _triton_local_cx_enabled(
            device_type=shard_state.amplitudes.device.type,
            dtype=str(shard_state.amplitudes.dtype).removeprefix("torch."),
            shape=(
                int(shard_state.amplitudes.shape[0]),
                int(shard_state.amplitudes.shape[1]),
            ),
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
            kernel_dispatch_evidence=(
                evidence.kernel_dispatch_evidence if evidence is not None else None
            ),
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

    from ....kernels.triton.statevector_adjoint import (
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
