"""PyTorch-native rank-local distributed statevector forward execution."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....core.ir import CircuitIR
from ....simulation.statevector.operations import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_diagonal_gate_eager,
    _apply_local_gate_eager,
    _basis_indices_for_wires,
    _basis_offset,
    _combine_gate_basis_blocks_eager,
    _combine_rank_pair_gate_eager,
    _wire_mask,
    _zero_basis_local_indices,
)
from ...distributed.identity import DistributedIdentity
from .environment import get_bool, mode
from .errors import FullStateMaterializationError
from .kernel_dispatch import (
    KernelDecision,
    KernelDispatchEvidence,
    select_triton_kernel,
)
from .models import (
    DistributedStatevectorPlan,
    StatevectorShardState,
)

_COMMUNICATION_LAYOUT_CACHE: dict[tuple[Any, ...], tuple[int, ...]] = {}
_COMMUNICATION_LAYOUT_CACHE_LIMIT = 128


def _is_diagonal_instruction(name: str) -> bool:
    return str(name).lower() in _DIAGONAL_STATEVECTOR_GATES


def _runtime_index_validation_enabled() -> bool:
    return os.getenv("FQ_STATEVECTOR_VALIDATE_INDICES", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _triton_local_1q_decision(*, supported: bool = True) -> KernelDecision:
    requested = os.getenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    return select_triton_kernel("local_1q", requested=requested, supported=supported)


def _triton_local_cx_decision(*, supported: bool = True) -> KernelDecision:
    return select_triton_kernel(
        "local_cx",
        requested=get_bool("FQ_STATEVECTOR_TRITON_LOCAL_CX", True),
        supported=supported,
    )


def _triton_local_cx_enabled() -> bool:
    return _triton_local_cx_decision().accelerated


def _triton_local_cx_segment_enabled(ir: CircuitIR | None = None) -> bool:
    raw = os.getenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "")
    if not raw:
        requested = bool(
            ir is not None
            and ir.metadata.get("statevector_dependency_schedule_changed", False)
        )
    else:
        requested = raw.strip().lower() in {
            "1",
            "true",
            "on",
            "yes",
        }
    return select_triton_kernel("local_cx_segment", requested=requested).accelerated


def _triton_transpose_1q_enabled() -> bool:
    requested = os.getenv(
        "FQ_STATEVECTOR_TRITON_TRANSPOSE_1Q", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    return select_triton_kernel("transpose_1q", requested=requested).accelerated


def _local_block_fusion_enabled() -> bool:
    if mode() == "portable":
        return False
    return os.getenv("FQ_STATEVECTOR_LOCAL_BLOCK_FUSION", "1").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _local_block_fusion_width() -> int:
    value = int(os.getenv("FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH", "3"))
    if value not in {2, 3, 4}:
        raise ValueError("FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH must be 2, 3, or 4")
    return value


def _cross_shard_cx_packing_enabled() -> bool:
    return os.getenv("FQ_STATEVECTOR_CROSS_SHARD_CX_PACK", "1").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _topology_aware_rank_bits_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_TOPOLOGY_AWARE_RANK_BITS", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _ket_checkpoint_mode() -> str:
    value = (
        os.getenv("FQ_STATEVECTOR_INTER_NODE_KET_CHECKPOINTS", "inter_node")
        .strip()
        .lower()
    )
    if value in {"0", "false", "off", "no"}:
        return "off"
    if value in {"all", "inter_node"}:
        return value
    return "inter_node"


def communication_aware_wire_layout(
    ir: CircuitIR,
    *,
    world_size: int,
    preferred_local_wires: Sequence[int] = (),
    optimization_target: str = "forward",
) -> tuple[CircuitIR, tuple[int, ...]]:
    """Relabel wires to minimize forward or full-training-step communication."""

    rank_bits = int(world_size).bit_length() - 1
    identity = tuple(range(ir.n_wires))
    if optimization_target not in {"forward", "training_step"}:
        raise ValueError("optimization_target must be forward or training_step")
    if world_size <= 1 or world_size & (world_size - 1) or rank_bits >= ir.n_wires:
        return ir, identity
    preferred = {int(wire) for wire in preferred_local_wires}
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE")
        or os.environ.get("NPROC_PER_NODE")
        or world_size
    )
    topology_aware = (
        _topology_aware_rank_bits_enabled() and world_size > local_world_size
    )
    cache_key = (
        ir.n_wires,
        int(world_size),
        local_world_size,
        topology_aware,
        optimization_target,
        tuple(sorted(preferred)),
        tuple(
            (
                instruction.name,
                tuple(map(int, instruction.wires)),
                tuple(
                    isinstance(value, torch.Tensor) and value.requires_grad
                    for value in instruction.params.values()
                ),
            )
            for instruction in ir.instructions
        ),
    )
    permutation = _COMMUNICATION_LAYOUT_CACHE.get(cache_key)

    if permutation is None:

        def candidate_cost(candidate: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
            sharded = set(candidate)
            local = [wire for wire in range(ir.n_wires) if wire not in sharded]
            physical = {logical: index for index, logical in enumerate(local)}
            cost = 0
            for instruction in ir.instructions:
                if not any(int(wire) in sharded for wire in instruction.wires):
                    continue
                width_weight = 2 ** max(0, len(instruction.wires) - 1)
                trainable = any(
                    isinstance(value, torch.Tensor) and value.requires_grad
                    for value in instruction.params.values()
                )
                execution_weight = (
                    (4 if trainable else 3)
                    if optimization_target == "training_step"
                    else 1
                )
                local_wires = [
                    int(wire) for wire in instruction.wires if int(wire) not in sharded
                ]
                # Subgroup exchange chunks must align to twice the largest local
                # wire mask.  Account for that directly: moving a sharded endpoint
                # can otherwise turn a bounded chunk into a complete local shard.
                alignment = 1
                if local_wires:
                    highest_position = max(
                        ir.n_wires - physical[wire] - 1 - rank_bits
                        for wire in local_wires
                    )
                    alignment = 1 << (highest_position + 1)
                cost += width_weight * execution_weight * alignment
            if sharded & preferred:
                cost += 1 << (ir.n_wires + rank_bits + 4)
            return cost, candidate

        selected_sharded_wires = min(
            combinations(range(ir.n_wires), rank_bits), key=candidate_cost
        )
        sharded_logical = set(selected_sharded_wires)
        wire_activity = [0] * ir.n_wires
        for instruction in ir.instructions:
            width_weight = 2 ** max(0, len(instruction.wires) - 1)
            trainable = any(
                isinstance(value, torch.Tensor) and value.requires_grad
                for value in instruction.params.values()
            )
            execution_weight = (
                (4 if trainable else 3) if optimization_target == "training_step" else 1
            )
            for wire in instruction.wires:
                wire_activity[int(wire)] += width_weight * execution_weight
        local_logical = [
            wire for wire in range(ir.n_wires) if wire not in sharded_logical
        ]
        mapping = [0] * ir.n_wires
        for physical, logical in enumerate(local_logical):
            mapping[logical] = physical
        ordered_sharded = sorted(
            sharded_logical,
            key=(
                lambda wire: (wire_activity[wire], wire) if topology_aware else (wire,)
            ),
        )
        for offset, logical in enumerate(ordered_sharded):
            mapping[logical] = ir.n_wires - rank_bits + offset
        permutation = tuple(mapping)
        if len(_COMMUNICATION_LAYOUT_CACHE) >= _COMMUNICATION_LAYOUT_CACHE_LIMIT:
            _COMMUNICATION_LAYOUT_CACHE.pop(next(iter(_COMMUNICATION_LAYOUT_CACHE)))
        _COMMUNICATION_LAYOUT_CACHE[cache_key] = permutation
    if permutation == identity:
        return ir, permutation

    def remap(wires: Sequence[int]) -> tuple[int, ...]:
        return tuple(permutation[int(wire)] for wire in wires)

    return (
        replace(
            ir,
            instructions=tuple(
                replace(instruction, wires=remap(instruction.wires))
                for instruction in ir.instructions
            ),
            observables=tuple(
                replace(observable, wires=remap(observable.wires))
                for observable in ir.observables
            ),
            measurements=tuple(
                replace(measurement, wires=remap(measurement.wires))
                for measurement in ir.measurements
            ),
            metadata={
                **ir.metadata,
                "statevector_logical_to_physical_wires": permutation,
                "statevector_layout_optimization_target": optimization_target,
            },
        ),
        permutation,
    )


def _basis_owner_rank(
    plan: DistributedStatevectorPlan,
    rank: int,
    wires: Sequence[int],
    basis: int,
) -> int:
    """Resolve a gate-basis owner from static rank-address bits."""

    wires = tuple(wires)
    owner = int(rank)
    width = len(wires)
    for position, wire in enumerate(wires):
        if wire not in plan.sharded_wires:
            continue
        rank_mask = 1 << (
            len(plan.sharded_wires) - plan.sharded_wires.index(int(wire)) - 1
        )
        bit = (int(basis) >> (width - position - 1)) & 1
        owner = (owner & ~rank_mask) | (bit * rank_mask)
    return owner


@dataclass
class StatevectorExchangeWorkspace:
    """Reusable receive buffers for graph-free production forward execution."""

    buffers: dict[tuple[int, str, torch.dtype], torch.Tensor] = field(
        default_factory=dict
    )
    allocation_count: int = 0
    reuse_count: int = 0
    reserved_bytes: int = 0
    pipeline_prefetch_count: int = 0
    pipelined_gate_count: int = 0
    subgroup_pipelined_gate_count: int = 0
    intra_node_message_count: int = 0
    intra_node_bytes: int = 0
    inter_node_message_count: int = 0
    inter_node_bytes: int = 0

    def record_peer_exchange(self, global_peer: int, byte_count: int) -> None:
        """Classify one rank's logical send bytes by torchrun node placement."""

        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE")
            or os.environ.get("NPROC_PER_NODE")
            or (dist.get_world_size() if dist.is_initialized() else 1)
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank // local_world_size == int(global_peer) // local_world_size:
            self.intra_node_message_count += 1
            self.intra_node_bytes += int(byte_count)
        else:
            self.inter_node_message_count += 1
            self.inter_node_bytes += int(byte_count)

    def acquire(
        self, like: torch.Tensor, *, slot: int = 0, leading_factor: int = 1
    ) -> torch.Tensor:
        shape = (like.shape[0] * int(leading_factor), *like.shape[1:])
        required = math.prod(shape)
        key = (int(slot), str(like.device), like.dtype)
        buffer = self.buffers.get(key)
        if buffer is None or buffer.numel() < required:
            previous_bytes = (
                buffer.numel() * buffer.element_size() if buffer is not None else 0
            )
            buffer = torch.empty(required, dtype=like.dtype, device=like.device)
            self.buffers[key] = buffer
            self.allocation_count += 1
            self.reserved_bytes += (
                buffer.numel() * buffer.element_size() - previous_bytes
            )
        else:
            self.reuse_count += 1
        return buffer[:required].view(shape)


@dataclass(frozen=True)
class TorchDistributedStatevectorResult:
    """Rank-local result from the PyTorch-native numerical executor."""

    shard_state: StatevectorShardState
    plan: DistributedStatevectorPlan
    backend: str
    local_gate_count: int
    distributed_gate_count: int
    communication_count: int
    communication_bytes: int
    peak_scratch_bytes: int
    local_diagonal_gate_count: int
    executed_distributed_segment_count: int
    fused_cross_shard_regions: int
    fused_cross_shard_gate_count: int
    cross_shard_fusion_enabled: bool
    exchange_workspace_allocation_count: int
    exchange_workspace_reuse_count: int
    exchange_workspace_reserved_bytes: int
    exchange_pipeline_prefetch_count: int
    exchange_pipelined_gate_count: int
    exchange_pipeline_enabled: bool
    exchange_subgroup_pipelined_gate_count: int
    intra_node_communication_count: int
    intra_node_communication_bytes: int
    inter_node_communication_count: int
    inter_node_communication_bytes: int
    wire_layout: str
    logical_to_physical_wires: tuple[int, ...]
    distributed_identity: DistributedIdentity
    kernel_dispatch_evidence: KernelDispatchEvidence = field(
        default_factory=KernelDispatchEvidence
    )
    persistent_inter_node_ket_checkpoints: tuple[
        tuple[int, int, torch.Tensor], ...
    ] = ()

    def full_state(self) -> torch.Tensor:
        raise FullStateMaterializationError(
            "distributed statevector results are rank-local; full-state "
            "materialization is forbidden on the production executor"
        )

    def summary(self) -> dict[str, Any]:
        accelerator = self.backend in {"nccl", "flagos"}
        sharded = self.plan.world_size > 1
        return {
            "executor": "pytorch_native_distributed_statevector_v1",
            "backend": self.backend,
            "distributed_identity": self.distributed_identity.to_dict(),
            "claim_evidence_type": (
                "accelerator_semantics" if accelerator else "development_semantics"
            ),
            "distribution_semantics": (
                "sharded_across_ranks" if sharded else "single_device_fast_path"
            ),
            "world_size": self.plan.world_size,
            "local_world_size": self.plan.local_world_size,
            "node_count": self.plan.node_count,
            "rank": self.shard_state.rank,
            "wire_layout": self.wire_layout,
            "logical_to_physical_wires": self.logical_to_physical_wires,
            "amplitude_basis_order": (
                "canonical_logical"
                if self.logical_to_physical_wires
                == tuple(range(len(self.logical_to_physical_wires)))
                else "internal_physical_requires_mapping"
            ),
            "rank_ownership": self.shard_state.summary(),
            "local_state_bytes": int(
                self.shard_state.amplitudes.numel()
                * self.shard_state.amplitudes.element_size()
            ),
            "peak_scratch_bytes": self.peak_scratch_bytes,
            "scratch_accounting": "peak_live_amplitude_tensors_including_output_buffer",
            "kernel_dispatch": self.kernel_dispatch_evidence.summary(),
            "local_gate_count": self.local_gate_count,
            "local_diagonal_gate_count": self.local_diagonal_gate_count,
            "distributed_gate_count": self.distributed_gate_count,
            "executed_distributed_segment_count": self.executed_distributed_segment_count,
            "fused_cross_shard_regions": self.fused_cross_shard_regions,
            "fused_cross_shard_gate_count": self.fused_cross_shard_gate_count,
            "cross_shard_fusion_enabled": self.cross_shard_fusion_enabled,
            "exchange_workspace_allocation_count": self.exchange_workspace_allocation_count,
            "exchange_workspace_reuse_count": self.exchange_workspace_reuse_count,
            "exchange_workspace_reserved_bytes": self.exchange_workspace_reserved_bytes,
            "exchange_pipeline_prefetch_count": self.exchange_pipeline_prefetch_count,
            "exchange_pipelined_gate_count": self.exchange_pipelined_gate_count,
            "exchange_pipeline_enabled": self.exchange_pipeline_enabled,
            "exchange_subgroup_pipelined_gate_count": self.exchange_subgroup_pipelined_gate_count,
            "communication_count": self.communication_count,
            "communication_bytes": self.communication_bytes,
            "intra_node_communication_count": self.intra_node_communication_count,
            "intra_node_communication_bytes": self.intra_node_communication_bytes,
            "inter_node_communication_count": self.inter_node_communication_count,
            "inter_node_communication_bytes": self.inter_node_communication_bytes,
            "communication_primitive": "torch.distributed.collective",
            "full_state_materialization": False,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": tuple(
                dict.fromkeys(
                    (
                        "forward_only_no_sharded_backward_or_optimizer",
                        *self.distributed_identity.claim_blockers_for(self.backend),
                    )
                )
            ),
        }


def _storage_global_indices(
    shard_state: StatevectorShardState,
    start: int,
    end: int,
    *,
    plan: DistributedStatevectorPlan,
) -> torch.Tensor:
    if shard_state.global_indices.numel():
        return shard_state.global_indices[start:end]
    local = torch.arange(
        start, end, dtype=torch.long, device=shard_state.amplitudes.device
    )
    if plan.distribution == "qubit_address_sharded":
        return (local << len(plan.sharded_wires)) | shard_state.rank
    return local + shard_state.shard.amplitude_start


def _owner_and_local(
    global_indices: torch.Tensor, *, plan: DistributedStatevectorPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    if plan.distribution == "qubit_address_sharded":
        rank_mask = plan.world_size - 1
        return global_indices & rank_mask, global_indices >> len(plan.sharded_wires)
    local_count = plan.shards[0].local_amplitudes
    return global_indices // local_count, global_indices.remainder(local_count)


def _independent_tensor_bytes(tensor: torch.Tensor, owner: torch.Tensor) -> int:
    """Return tensor bytes only when it does not alias the owner's storage."""

    if tensor.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr():
        return 0
    return tensor.numel() * tensor.element_size()


def _wait_for_exchange(request: Any, device: torch.device) -> None:
    """Wait through a route whose stream-ordering semantics are verified."""

    if device.type == "cuda" and hasattr(request, "block_current_stream"):
        request.block_current_stream()
        return
    request.wait()


def _vectorized_local_gate(
    shard_state: StatevectorShardState,
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
    chunk_amplitudes: int,
    output: torch.Tensor | None = None,
    kernel_dispatch_evidence: KernelDispatchEvidence | None = None,
) -> tuple[StatevectorShardState, int]:
    """Apply owned outputs in tensor chunks without amplitude Python loops."""

    wires = tuple(int(wire) for wire in wires)
    gate_dim = 2 ** len(wires)
    matrix = matrix.to(shard_state.amplitudes)
    rank_bits = len(plan.sharded_wires)
    triton_supported = bool(
        gate_dim == 2
        and shard_state.amplitudes.device.type == "cuda"
        and shard_state.amplitudes.dtype == torch.complex64
        and shard_state.amplitudes.is_contiguous()
    )
    triton_decision = _triton_local_1q_decision(supported=triton_supported)
    if gate_dim == 2 and kernel_dispatch_evidence is not None:
        kernel_dispatch_evidence.record(triton_decision)
    if triton_decision.accelerated:
        from ....simulation.triton_kernels.statevector_gates import (
            apply_complex64_local_1q,
        )

        bit_position = plan.n_wires - wires[0] - 1 - rank_bits
        amplitudes = apply_complex64_local_1q(
            shard_state.amplitudes,
            matrix,
            bit_position=bit_position,
            output=output,
        )
        output_bytes = amplitudes.numel() * amplitudes.element_size()
        return (
            StatevectorShardState(
                rank=shard_state.rank,
                shard=shard_state.shard,
                amplitudes=amplitudes,
                global_indices=shard_state.global_indices,
            ),
            output_bytes,
        )
    amplitudes, peak = _apply_local_gate_eager(
        shard_state.amplitudes,
        matrix,
        wires,
        n_wires=plan.n_wires,
        rank_bits=rank_bits,
        chunk_amplitudes=chunk_amplitudes,
        output=output,
        validate_indices=_runtime_index_validation_enabled(),
    )
    return (
        StatevectorShardState(
            rank=shard_state.rank,
            shard=shard_state.shard,
            amplitudes=amplitudes,
            global_indices=shard_state.global_indices,
        ),
        peak,
    )


def _vectorized_local_cx_gate(
    shard_state: StatevectorShardState,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
    output: torch.Tensor | None = None,
) -> tuple[StatevectorShardState, int]:
    """Apply a fully local CNOT with no basis-index or matrix temporaries."""

    from ....simulation.triton_kernels.statevector_gates import (
        apply_complex64_local_cx_inplace,
    )

    control, target = (int(wire) for wire in wires)
    rank_bits = len(plan.sharded_wires)
    result = torch.empty_like(shard_state.amplitudes) if output is None else output
    if result.data_ptr() != shard_state.amplitudes.data_ptr():
        result.copy_(shard_state.amplitudes)
    apply_complex64_local_cx_inplace(
        result,
        control_bit_position=plan.n_wires - control - 1 - rank_bits,
        target_bit_position=plan.n_wires - target - 1 - rank_bits,
    )
    return (
        StatevectorShardState(
            rank=shard_state.rank,
            shard=shard_state.shard,
            amplitudes=result,
            global_indices=shard_state.global_indices,
        ),
        result.numel() * result.element_size(),
    )


def _vectorized_local_diagonal_gate(
    shard_state: StatevectorShardState,
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
    chunk_amplitudes: int,
    output: torch.Tensor | None = None,
) -> tuple[StatevectorShardState, int]:
    """Apply a diagonal gate rank-locally, including on sharded address bits."""

    diagonal = torch.diagonal(matrix.to(shard_state.amplitudes), dim1=-2, dim2=-1)
    out = torch.empty_like(shard_state.amplitudes) if output is None else output
    output_bytes = out.numel() * out.element_size()
    peak = output_bytes
    local_count = shard_state.shard.local_amplitudes
    for start in range(0, local_count, chunk_amplitudes):
        end = min(local_count, start + chunk_amplitudes)
        global_indices = _storage_global_indices(shard_state, start, end, plan=plan)
        _, factor_bytes = _apply_diagonal_gate_eager(
            shard_state.amplitudes[:, start:end],
            diagonal,
            global_indices,
            wires,
            n_wires=plan.n_wires,
            output=out[:, start:end],
        )
        peak = max(peak, output_bytes + factor_bytes)
    return (
        StatevectorShardState(
            rank=shard_state.rank,
            shard=shard_state.shard,
            amplitudes=out,
            global_indices=shard_state.global_indices,
        ),
        peak,
    )


def _vectorized_pair_exchange_gate(
    shard_state: StatevectorShardState,
    matrix: torch.Tensor,
    wire: int,
    *,
    plan: DistributedStatevectorPlan,
    chunk_amplitudes: int,
    process_group: Any | None = None,
    workspace: StatevectorExchangeWorkspace | None = None,
    pipeline: bool = True,
    output: torch.Tensor | None = None,
) -> tuple[StatevectorShardState, int, int, int]:
    """Apply a one-sharded-wire gate through chunked rank-pair exchange."""

    position = plan.sharded_wires.index(int(wire))
    rank_mask = 1 << (len(plan.sharded_wires) - position - 1)
    peer = shard_state.rank ^ rank_mask
    global_peer = (
        dist.get_global_rank(process_group, peer) if process_group is not None else peer
    )
    rank_basis = (shard_state.rank >> (len(plan.sharded_wires) - position - 1)) & 1
    matrix = matrix.to(shard_state.amplitudes)
    out = torch.empty_like(shard_state.amplitudes) if output is None else output
    output_bytes = out.numel() * out.element_size()
    local_count = shard_state.shard.local_amplitudes
    communication_count = communication_bytes = peak = 0
    chunks = tuple(
        (start, min(local_count, start + chunk_amplitudes))
        for start in range(0, local_count, chunk_amplitudes)
    )

    def post(
        chunk_index: int,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, list[Any], int]:
        start, end = chunks[chunk_index]
        local = shard_state.amplitudes[:, start:end].contiguous()
        if plan.world_size == 2:
            gathered = (
                workspace.acquire(
                    local,
                    slot=chunk_index % 2 if pipeline else 0,
                    leading_factor=2,
                )
                if workspace is not None
                else torch.empty(
                    (local.shape[0] * 2, *local.shape[1:]),
                    dtype=local.dtype,
                    device=local.device,
                )
            )
            all_gather = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
            request = all_gather(gathered, local, group=process_group, async_op=True)
            bsz = local.shape[0]
            remote_start = (1 - shard_state.rank) * bsz
            remote = gathered[remote_start : remote_start + bsz]
            return (
                start,
                end,
                local,
                remote,
                [request],
                gathered.numel() * gathered.element_size(),
            )
        remote = (
            workspace.acquire(local, slot=chunk_index % 2 if pipeline else 0)
            if workspace is not None
            else torch.empty_like(local)
        )
        operations = [
            dist.P2POp(dist.isend, local, global_peer, process_group, chunk_index),
            dist.P2POp(dist.irecv, remote, global_peer, process_group, chunk_index),
        ]
        return (
            start,
            end,
            local,
            remote,
            list(dist.batch_isend_irecv(operations)),
            remote.numel() * remote.element_size(),
        )

    current = post(0)
    pipelined = pipeline and workspace is not None and len(chunks) > 1
    if pipelined:
        workspace.pipelined_gate_count += 1
    for chunk_index in range(len(chunks)):
        start, end, local, remote, requests, exchange_storage_bytes = current
        for request in requests:
            _wait_for_exchange(request, local.device)
        following = None
        if pipelined and chunk_index + 1 < len(chunks):
            following = post(chunk_index + 1)
            workspace.pipeline_prefetch_count += 1
        updated, numeric_scratch_bytes = _combine_rank_pair_gate_eager(
            local, remote, matrix, rank_basis=rank_basis
        )
        out[:, start:end] = updated
        communication_count += 1
        sent_bytes = local.numel() * local.element_size()
        communication_bytes += sent_bytes
        if workspace is not None:
            workspace.record_peer_exchange(global_peer, sent_bytes)
        peak = max(
            peak,
            output_bytes
            + _independent_tensor_bytes(local, shard_state.amplitudes)
            + exchange_storage_bytes
            + numeric_scratch_bytes,
        )
        if chunk_index + 1 < len(chunks):
            current = following if following is not None else post(chunk_index + 1)
    return (
        StatevectorShardState(
            rank=shard_state.rank,
            shard=shard_state.shard,
            amplitudes=out,
            global_indices=shard_state.global_indices,
        ),
        communication_count,
        communication_bytes,
        peak,
    )


def _vectorized_cross_shard_cx(
    shard_state: StatevectorShardState,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
    chunk_amplitudes: int,
    process_group: Any | None = None,
    workspace: StatevectorExchangeWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> tuple[StatevectorShardState, int, int, int]:
    """Apply a CX touching exactly one rank bit with minimal communication."""

    control, target = (int(wire) for wire in wires)
    control_sharded = control in plan.sharded_wires
    target_sharded = target in plan.sharded_wires
    if control_sharded == target_sharded:
        raise ValueError("specialized cross-shard CX requires exactly one sharded wire")

    if control_sharded:
        position = plan.sharded_wires.index(control)
        rank_shift = len(plan.sharded_wires) - position - 1
        rank_control = (shard_state.rank >> rank_shift) & 1
        if rank_control == 0:
            amplitudes = (
                shard_state.amplitudes.clone()
                if output is None
                else output.copy_(shard_state.amplitudes)
            )
            return (
                replace(shard_state, amplitudes=amplitudes),
                0,
                0,
                (amplitudes.numel() * amplitudes.element_size()),
            )
        x_matrix = torch.tensor(
            ((0, 1), (1, 0)),
            dtype=shard_state.amplitudes.dtype,
            device=shard_state.amplitudes.device,
        )
        return_state, scratch = _vectorized_local_gate(
            shard_state,
            x_matrix,
            (target,),
            plan=plan,
            chunk_amplitudes=chunk_amplitudes,
            output=output,
        )
        return return_state, 0, 0, scratch

    position = plan.sharded_wires.index(target)
    rank_shift = len(plan.sharded_wires) - position - 1
    peer = shard_state.rank ^ (1 << rank_shift)
    global_peer = (
        dist.get_global_rank(process_group, peer) if process_group is not None else peer
    )
    out = (
        shard_state.amplitudes.clone()
        if output is None
        else output.copy_(shard_state.amplitudes)
    )
    output_bytes = out.numel() * out.element_size()
    rank_bits = len(plan.sharded_wires)
    control_position = plan.n_wires - control - 1 - rank_bits
    control_mask = 1 << control_position
    active_count = shard_state.shard.local_amplitudes // 2
    active_chunk = max(1, chunk_amplitudes)
    communication_count = communication_bytes = 0
    peak = output_bytes
    use_triton_pack = (
        shard_state.amplitudes.device.type == "cuda"
        and shard_state.amplitudes.dtype == torch.complex64
        and shard_state.amplitudes.is_contiguous()
        and out.is_contiguous()
    )
    if use_triton_pack:
        from ....simulation.triton_kernels.statevector_gates import (
            pack_complex64_control_one,
            unpack_complex64_control_one,
        )
    for chunk_index, start in enumerate(range(0, active_count, active_chunk)):
        end = min(active_count, start + active_chunk)
        if use_triton_pack:
            local = pack_complex64_control_one(
                shard_state.amplitudes,
                bit_position=control_position,
                compressed_start=start,
                compressed_end=end,
            )
            indices = None
        else:
            indices = _zero_basis_local_indices(
                start,
                end,
                (control,),
                n_wires=plan.n_wires,
                rank_bits=rank_bits,
                device=out.device,
            )
            indices |= control_mask
            local = shard_state.amplitudes[:, indices].contiguous()
        remote = (
            workspace.acquire(local, slot=0)
            if workspace is not None
            else torch.empty_like(local)
        )
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, local, global_peer, process_group, chunk_index),
                dist.P2POp(dist.irecv, remote, global_peer, process_group, chunk_index),
            ]
        )
        for request in requests:
            _wait_for_exchange(request, local.device)
        if use_triton_pack:
            unpack_complex64_control_one(
                remote,
                out,
                bit_position=control_position,
                compressed_start=start,
            )
        else:
            assert indices is not None
            out[:, indices] = remote
        communication_count += 1
        sent_bytes = local.numel() * local.element_size()
        communication_bytes += sent_bytes
        if workspace is not None:
            workspace.record_peer_exchange(global_peer, sent_bytes)
        peak = max(
            peak,
            output_bytes
            + local.numel() * local.element_size()
            + remote.numel() * remote.element_size()
            + (0 if indices is None else indices.numel() * indices.element_size()),
        )
    return (
        replace(shard_state, amplitudes=out),
        communication_count,
        communication_bytes,
        peak,
    )


def _vectorized_subgroup_exchange_gate(
    shard_state: StatevectorShardState,
    matrix: torch.Tensor,
    wires: Sequence[int],
    *,
    plan: DistributedStatevectorPlan,
    chunk_amplitudes: int,
    process_group: Any | None = None,
    workspace: StatevectorExchangeWorkspace | None = None,
    pipeline: bool = True,
    output: torch.Tensor | None = None,
) -> tuple[StatevectorShardState, int, int, int]:
    """Apply a cross-shard gate using chunked subgroup all-to-all exchange."""

    wires = tuple(int(wire) for wire in wires)
    sharded = tuple(wire for wire in wires if wire in plan.sharded_wires)
    local_wires = tuple(wire for wire in wires if wire not in plan.sharded_wires)
    rank_masks = tuple(
        1 << (len(plan.sharded_wires) - plan.sharded_wires.index(wire) - 1)
        for wire in sharded
    )
    peers = tuple(
        shard_state.rank
        ^ sum(mask for bit, mask in enumerate(rank_masks) if code & (1 << bit))
        for code in range(1, 2 ** len(rank_masks))
    )
    local_count = shard_state.shard.local_amplitudes
    gate_dim = 2 ** len(wires)
    local_masks = tuple(
        _wire_mask(plan.n_wires, wire) >> len(plan.sharded_wires)
        for wire in local_wires
    )
    alignment = 2 * max(local_masks) if local_masks else 1
    scratch_terms = len(peers) + gate_dim
    max_substate_chunk = max(1, (local_count * plan.world_size - 1) // scratch_terms)
    requested = min(chunk_amplitudes, max_substate_chunk)
    chunk_amplitudes = max(alignment, (requested // alignment) * alignment)
    offsets = torch.tensor(
        [_basis_offset(plan.n_wires, wires, basis) for basis in range(gate_dim)],
        dtype=torch.long,
        device=shard_state.amplitudes.device,
    )
    rank_bits = len(plan.sharded_wires)
    local_offsets = offsets >> rank_bits
    local_clear_mask = ~sum(
        _wire_mask(plan.n_wires, wire) >> rank_bits for wire in local_wires
    )
    clear_mask = ~sum(_wire_mask(plan.n_wires, wire) for wire in wires)
    matrix = matrix.to(shard_state.amplitudes)
    out = torch.empty_like(shard_state.amplitudes) if output is None else output
    output_bytes = out.numel() * out.element_size()
    communication_count = communication_bytes = peak = 0
    chunks = tuple(
        (start, min(local_count, start + chunk_amplitudes))
        for start in range(0, local_count, chunk_amplitudes)
    )

    def post(
        chunk_index: int,
    ) -> tuple[int, int, torch.Tensor, dict[int, torch.Tensor], list[Any]]:
        start, end = chunks[chunk_index]
        local = shard_state.amplitudes[:, start:end].contiguous()
        received = {
            peer: (
                workspace.acquire(
                    local,
                    slot=((chunk_index % 2) * len(peers) + slot if pipeline else slot),
                )
                if workspace is not None
                else torch.empty_like(local)
            )
            for slot, peer in enumerate(peers)
        }
        operations: list[dist.P2POp] = []
        for peer in peers:
            global_peer = (
                dist.get_global_rank(process_group, peer)
                if process_group is not None
                else peer
            )
            operations.extend(
                (
                    dist.P2POp(
                        dist.isend, local, global_peer, process_group, chunk_index
                    ),
                    dist.P2POp(
                        dist.irecv,
                        received[peer],
                        global_peer,
                        process_group,
                        chunk_index,
                    ),
                )
            )
        return (
            start,
            end,
            local,
            received,
            list(dist.batch_isend_irecv(operations)),
        )

    current = post(0)
    pipelined = pipeline and workspace is not None and len(chunks) > 1
    if pipelined:
        workspace.pipelined_gate_count += 1
        workspace.subgroup_pipelined_gate_count += 1
    for chunk_index in range(len(chunks)):
        start, end, local, received, requests = current
        for request in requests:
            _wait_for_exchange(request, local.device)
        following = None
        if pipelined and chunk_index + 1 < len(chunks):
            following = post(chunk_index + 1)
            workspace.pipeline_prefetch_count += 1
        sources = {shard_state.rank: local, **received}
        global_out = _storage_global_indices(shard_state, start, end, plan=plan)
        if plan.distribution == "qubit_address_sharded":
            local_out = torch.arange(
                start, end, dtype=torch.long, device=global_out.device
            )
            local_bases = local_out & local_clear_mask
            required = None
        else:
            local_bases = None
            required = (global_out & clear_mask)[:, None] | offsets[None, :]
        output_basis = _basis_indices_for_wires(
            global_out, n_wires=plan.n_wires, wires=wires
        )
        basis_inputs = []
        for basis in range(gate_dim):
            if local_bases is not None:
                local_required = local_bases | local_offsets[basis]
            else:
                assert required is not None
                _, local_required = _owner_and_local(required[:, basis], plan=plan)
            owner = _basis_owner_rank(plan, shard_state.rank, wires, basis)
            local_required = local_required - start
            if _runtime_index_validation_enabled() and (
                int(local_required.min()) < 0
                or int(local_required.max()) >= end - start
            ):
                raise ValueError(
                    "exchange chunk is not aligned to the gate's local wires"
                )
            basis_inputs.append(sources[owner][:, local_required])
        updated_output, numeric_scratch_bytes = _combine_gate_basis_blocks_eager(
            basis_inputs, matrix, output_basis
        )
        out[:, start:end] = updated_output
        communication_count += len(peers)
        sent_bytes = local.numel() * local.element_size()
        communication_bytes += len(peers) * sent_bytes
        if workspace is not None:
            for peer in peers:
                global_peer = (
                    dist.get_global_rank(process_group, peer)
                    if process_group is not None
                    else peer
                )
                workspace.record_peer_exchange(global_peer, sent_bytes)
        peak = max(
            peak,
            output_bytes
            + _independent_tensor_bytes(local, shard_state.amplitudes)
            + sum(value.numel() * value.element_size() for value in received.values())
            + numeric_scratch_bytes,
        )
        if chunk_index + 1 < len(chunks):
            current = following if following is not None else post(chunk_index + 1)
    return (
        StatevectorShardState(
            rank=shard_state.rank,
            shard=shard_state.shard,
            amplitudes=out,
            global_indices=shard_state.global_indices,
        ),
        communication_count,
        communication_bytes,
        peak,
    )
