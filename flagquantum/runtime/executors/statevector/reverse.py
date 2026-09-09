"""Native PyTorch reverse mode over rank-local statevector shards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....compute import resolve_platform_device
from ....core.ir import CircuitIR, ensure_circuit_ir
from ...builder_compilation import detached_ir_snapshot
from .checkpointing import StatevectorCheckpointPolicy, resolve_checkpoint_policy
from .forward import communication_aware_wire_layout
from .forward_executor import execute_torch_distributed_statevector
from .layout import (
    schedule_statevector_dependency_dag,
)
from .planning import plan_distributed_statevector
from .reverse_support import (
    BackwardExecutionEvidence,
    ParameterGradientOwnership,
    _bind_parameters,
    _persistent_wire_layout_enabled,
    _reverse_chunk_amplitudes,
)


def _communication_aware_layout_enabled() -> bool:
    return os.getenv("FQ_STATEVECTOR_COMM_AWARE_LAYOUT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _communication_aware_wire_layout(
    ir: CircuitIR, *, observable_wires: tuple[int, ...], world_size: int
) -> tuple[CircuitIR, tuple[int, ...], tuple[int, ...]]:
    """Use the shared alignment-aware layout for differentiable execution."""

    if not _communication_aware_layout_enabled():
        identity = tuple(range(ir.n_wires))
        return ir, observable_wires, identity
    remapped, permutation = communication_aware_wire_layout(
        ir,
        world_size=world_size,
        preferred_local_wires=observable_wires,
        optimization_target="training_step",
    )
    return remapped, tuple(permutation[wire] for wire in observable_wires), permutation


def _parameter_layout(
    ir: CircuitIR,
) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[int, str, int], ...], tuple[int, ...]]:
    parameters: list[torch.Tensor] = []
    identities: dict[int, int] = {}
    slots: list[tuple[int, str, int]] = []
    occurrences: list[int] = []
    for instruction_index, instruction in enumerate(ir.instructions):
        for name, value in instruction.params.items():
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                continue
            identity = id(value)
            parameter_index = identities.get(identity)
            if parameter_index is None:
                parameter_index = len(parameters)
                identities[identity] = parameter_index
                parameters.append(value)
                occurrences.append(0)
            occurrences[parameter_index] += 1
            slots.append((instruction_index, str(name), parameter_index))
    if not parameters:
        raise ValueError("sharded reverse mode requires at least one trainable tensor")
    return tuple(parameters), tuple(slots), tuple(occurrences)


@dataclass(frozen=True)
class TorchDistributedStatevectorGradientResult:
    value: torch.Tensor
    parameters: tuple[torch.Tensor, ...]
    ownership: tuple[ParameterGradientOwnership, ...]
    checkpoint_policy: StatevectorCheckpointPolicy
    world_size: int
    local_world_size: int
    node_count: int
    backend: str
    rank: int
    local_amplitudes: int
    total_amplitudes: int
    logical_to_physical_wires: tuple[int, ...]
    observable_wires: tuple[int, ...]
    layout_optimization_target: str
    backward_evidence: BackwardExecutionEvidence

    def backward(self) -> None:
        self.value.backward()

    def summary(self) -> dict[str, Any]:
        ready = self.backward_evidence.status == "completed"
        backward_semantics = (
            "sharded_across_ranks"
            if ready and self.world_size > 1
            else "single_device_fast_path" if ready else self.backward_evidence.status
        )
        ownership = tuple(
            {
                "parameter": item.parameter,
                "owner_rank": None,
                "occurrence_count": item.occurrence_count,
                "reduction": item.reduction if ready else "pending",
                "reduction_count": (
                    self.backward_evidence.reduction_counts[index] if ready else 0
                ),
                "participating_ranks": (
                    self.backward_evidence.participating_ranks if ready else ()
                ),
                "distribution": (
                    "replicated_after_all_reduce"
                    if ready and self.world_size > 1
                    else "local" if ready else "pending"
                ),
            }
            for index, item in enumerate(self.ownership)
        )
        return {
            "executor": "pytorch_native_sharded_statevector_reverse_v1",
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "backend": self.backend,
            "logical_to_physical_wires": self.logical_to_physical_wires,
            "layout_optimization_target": self.layout_optimization_target,
            "gradient_method": "statevector_adjoint",
            "observable_profile": "sum_of_single_wire_pauli_z_terms",
            "observable_wires": self.observable_wires,
            "distribution_semantics": (
                "sharded_across_ranks"
                if self.world_size > 1
                else "single_device_fast_path"
            ),
            "rank_ownership": {
                "rank": self.rank,
                "local_amplitudes": self.local_amplitudes,
                "total_amplitudes": self.total_amplitudes,
                "full_state_owned": self.local_amplitudes == self.total_amplitudes,
            },
            "forward_distribution_semantics": (
                "sharded_across_ranks"
                if self.world_size > 1
                else "single_device_fast_path"
            ),
            "backward_distribution_semantics": backward_semantics,
            "backward_status": self.backward_evidence.status,
            "backward_error": self.backward_evidence.error,
            "parameter_gradient_ready": ready,
            "gradient_distribution": (
                "replicated_after_all_reduce"
                if ready and self.world_size > 1
                else "local" if ready else "pending"
            ),
            "gradient_reduction": (
                "all_reduce_sum"
                if ready and self.world_size > 1
                else "local" if ready else "pending"
            ),
            "parameter_ownership": ownership,
            "checkpoint_policy": self.checkpoint_policy.__dict__,
            "local_state_bytes": self.local_amplitudes * self.value.element_size() * 2,
            "peak_backward_scratch_bytes": self.backward_evidence.peak_scratch_bytes,
            "backward_communication_count": self.backward_evidence.communication_count,
            "backward_communication_bytes": self.backward_evidence.communication_bytes,
            "gradient_collective_count": (
                self.backward_evidence.gradient_collective_count
            ),
            "gradient_collective_bytes": (
                self.backward_evidence.gradient_collective_bytes
            ),
            "async_gradient_collective_count": (
                self.backward_evidence.async_gradient_collective_count
            ),
            "overlapped_gradient_collective_count": (
                self.backward_evidence.overlapped_gradient_collective_count
            ),
            "persistent_layout_enabled": self.backward_evidence.persistent_layout_enabled,
            "persistent_layout_swap_count": (
                self.backward_evidence.persistent_layout_swap_count
            ),
            "persistent_layout_swap_bytes": (
                self.backward_evidence.persistent_layout_swap_bytes
            ),
            "persistent_layout_buffer_allocation_count": (
                self.backward_evidence.persistent_layout_buffer_allocation_count
            ),
            "persistent_layout_buffer_reserved_bytes": (
                self.backward_evidence.persistent_layout_buffer_reserved_bytes
            ),
            "persistent_inplace_local_enabled": (
                self.backward_evidence.persistent_inplace_local_enabled
            ),
            "analytic_rotation_derivative_count": (
                self.backward_evidence.analytic_rotation_derivative_count
            ),
            "fused_parameter_adjoint_count": (
                self.backward_evidence.fused_parameter_adjoint_count
            ),
            "backward_intra_node_communication_count": (
                self.backward_evidence.intra_node_communication_count
            ),
            "backward_intra_node_communication_bytes": (
                self.backward_evidence.intra_node_communication_bytes
            ),
            "backward_inter_node_communication_count": (
                self.backward_evidence.inter_node_communication_count
            ),
            "backward_inter_node_communication_bytes": (
                self.backward_evidence.inter_node_communication_bytes
            ),
            "saved_forward_state_reused": self.backward_evidence.saved_forward_state_reused,
            "backward_exchange_chunk_amplitudes": (
                self.backward_evidence.exchange_chunk_amplitudes
            ),
            "backward_exchange_chunk_bytes": self.backward_evidence.exchange_chunk_bytes,
            "backward_exchange_workspace_allocation_count": (
                self.backward_evidence.exchange_workspace_allocation_count
            ),
            "backward_exchange_workspace_reuse_count": (
                self.backward_evidence.exchange_workspace_reuse_count
            ),
            "backward_exchange_workspace_reserved_bytes": (
                self.backward_evidence.exchange_workspace_reserved_bytes
            ),
            "backward_exchange_pipeline_prefetch_count": (
                self.backward_evidence.exchange_pipeline_prefetch_count
            ),
            "kernel_dispatch": self.backward_evidence.kernel_dispatch_evidence.summary(),
            "backward_uses_full_state_replay": False,
            "full_state_materialization": False,
            "optimizer_update_ready": False,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": ("sharded_optimizer_update_pending",),
        }


class _ShardedStatevectorExpectation(torch.autograd.Function):
    """Rematerializing custom-autograd boundary; no global state is saved."""

    @staticmethod
    def forward(
        ctx: Any,
        ir: CircuitIR,
        slots: tuple[tuple[int, str, int], ...],
        policy: StatevectorCheckpointPolicy,
        observable_wires: tuple[int, ...],
        device: torch.device,
        process_group: Any | None,
        evidence: BackwardExecutionEvidence,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        from .reverse_adjoint import _local_expectation_z_sum

        ctx.slots = slots
        ctx.policy = policy
        ctx.observable_wires = observable_wires
        ctx.device = device
        ctx.process_group = process_group
        ctx.evidence = evidence
        ctx.save_for_backward(*parameters)
        bound = _bind_parameters(detached_ir_snapshot(ir), slots, parameters)
        # Builder slot mappings are task-local and are cleared after forward.
        # Retain the materialized IR so backward never dereferences cleared slots.
        ctx.ir = bound
        dtype = (
            torch.complex128
            if any(parameter.dtype == torch.float64 for parameter in parameters)
            else torch.complex64
        )
        result = execute_torch_distributed_statevector(
            bound,
            device=device,
            dtype=dtype,
            process_group=process_group,
            persistent_wire_layout=(
                _persistent_wire_layout_enabled()
                and policy.strategy == "reversible_adjoint"
            ),
        )
        ctx.forward_shard_state = (
            result.shard_state if policy.strategy == "reversible_adjoint" else None
        )
        ctx.forward_inter_node_ket_checkpoints = (
            result.persistent_inter_node_ket_checkpoints
            if policy.strategy == "reversible_adjoint"
            else ()
        )
        value = _local_expectation_z_sum(
            result.shard_state,
            plan=result.plan,
            n_wires=ir.n_wires,
            wires=tuple(
                result.logical_to_physical_wires[wire] for wire in observable_wires
            ),
        )
        if dist.is_initialized() and dist.get_world_size(process_group) > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        from .reverse_adjoint import _explicit_sharded_adjoint

        evidence = ctx.evidence
        evidence.status = "running"
        evidence.error = None
        evidence.participating_ranks = ()
        evidence.communication_count = 0
        evidence.communication_bytes = 0
        evidence.gradient_collective_count = 0
        evidence.gradient_collective_bytes = 0
        evidence.async_gradient_collective_count = 0
        evidence.overlapped_gradient_collective_count = 0
        evidence.persistent_layout_enabled = False
        evidence.persistent_layout_swap_count = 0
        evidence.persistent_layout_swap_bytes = 0
        evidence.persistent_layout_buffer_allocation_count = 0
        evidence.persistent_layout_buffer_reserved_bytes = 0
        evidence.persistent_inplace_local_enabled = False
        evidence.analytic_rotation_derivative_count = 0
        evidence.fused_parameter_adjoint_count = 0
        evidence.peak_scratch_bytes = 0
        evidence.saved_forward_state_reused = False
        evidence.exchange_workspace_allocation_count = 0
        evidence.exchange_workspace_reuse_count = 0
        evidence.exchange_workspace_reserved_bytes = 0
        evidence.exchange_pipeline_prefetch_count = 0
        evidence.intra_node_communication_count = 0
        evidence.intra_node_communication_bytes = 0
        evidence.inter_node_communication_count = 0
        evidence.inter_node_communication_bytes = 0
        evidence.exchange_chunk_amplitudes = _reverse_chunk_amplitudes()
        complex_element_bytes = (
            16
            if any(parameter.dtype == torch.float64 for parameter in ctx.saved_tensors)
            else 8
        )
        evidence.exchange_chunk_bytes = (
            evidence.exchange_chunk_amplitudes * complex_element_bytes
        )
        evidence.reduction_counts[:] = [0] * len(evidence.reduction_counts)
        try:
            with torch.enable_grad():
                gradients = _explicit_sharded_adjoint(
                    ctx.ir,
                    ctx.slots,
                    ctx.saved_tensors,
                    observable_wires=ctx.observable_wires,
                    device=ctx.device,
                    policy=ctx.policy,
                    process_group=ctx.process_group,
                    evidence=evidence,
                    saved_final_state=ctx.forward_shard_state,
                    saved_inter_node_ket_checkpoints=(
                        ctx.forward_inter_node_ket_checkpoints
                    ),
                )
            ctx.forward_shard_state = None
            ctx.forward_inter_node_ket_checkpoints = ()
            world_size = (
                dist.get_world_size(ctx.process_group) if dist.is_initialized() else 1
            )
            if world_size > 1:
                rank = dist.get_rank(ctx.process_group)
                local_rank = torch.tensor([rank], dtype=torch.int64, device=ctx.device)
                gathered = [torch.empty_like(local_rank) for _ in range(world_size)]
                dist.all_gather(gathered, local_rank, group=ctx.process_group)
                evidence.communication_count += 1
                evidence.communication_bytes += (
                    local_rank.numel() * local_rank.element_size()
                )
                evidence.participating_ranks = tuple(
                    sorted(int(item.item()) for item in gathered)
                )
            else:
                evidence.participating_ranks = (0,)
            scaled = tuple(item * grad_output.to(item.device) for item in gradients)
            for item in evidence.ownership:
                item.reduction = "all_reduce_sum" if world_size > 1 else "local"
                item.participating_ranks = evidence.participating_ranks
                item.distribution = (
                    "replicated_after_all_reduce" if world_size > 1 else "local"
                )
            evidence.status = "completed"
        except Exception as error:
            evidence.status = "failed"
            evidence.error = f"{type(error).__name__}: {error}"
            for item in evidence.ownership:
                item.reduction = "failed"
                item.distribution = "failed"
            raise
        return (None, None, None, None, None, None, None, *scaled)


def execute_torch_distributed_statevector_reverse(
    circuit_or_ir: Any,
    *,
    observable_wire: int = 0,
    observable_wires: Sequence[int] | None = None,
    checkpoint_policy: StatevectorCheckpointPolicy | None = None,
    device: torch.device | str | None = None,
    process_group: Any | None = None,
) -> TorchDistributedStatevectorGradientResult:
    """Return a differentiable global sum of single-wire Z expectations."""

    ir = ensure_circuit_ir(circuit_or_ir)
    selected_observable_wires = (
        (observable_wire,) if observable_wires is None else tuple(observable_wires)
    )
    if not selected_observable_wires:
        raise ValueError("observable_wires must contain at least one wire")
    if any(type(wire) is not int for wire in selected_observable_wires):
        raise TypeError("observable wires must be integers")
    outside = tuple(
        wire for wire in selected_observable_wires if wire < 0 or wire >= ir.n_wires
    )
    if outside:
        raise ValueError(f"observable wires {outside} are outside the circuit")
    backend = (
        dist.get_backend(process_group) if dist.is_initialized() else "single_process"
    )
    requested_policy = checkpoint_policy or StatevectorCheckpointPolicy()
    world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
    rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    if process_group is not None:
        local_world_size = world_size
    else:
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE")
            or os.environ.get("NPROC_PER_NODE")
            or world_size
        )
    (
        execution_ir,
        execution_observable_wires,
        execution_mapping,
    ) = _communication_aware_wire_layout(
        ir,
        observable_wires=selected_observable_wires,
        world_size=world_size,
    )
    if _persistent_wire_layout_enabled() and world_size > 1:
        execution_ir = schedule_statevector_dependency_dag(
            execution_ir, world_size=world_size
        )
    parameters, slots, occurrences = _parameter_layout(execution_ir)
    if device is None:
        device = (
            resolve_platform_device("cuda")
            if backend == "nccl"
            else parameters[0].device
        )
    resolved_device = torch.device(device)
    complex_bytes = (
        16 if any(parameter.dtype == torch.float64 for parameter in parameters) else 8
    )
    plan = plan_distributed_statevector(
        execution_ir,
        world_size=world_size,
        local_world_size=local_world_size,
        complex_bytes=complex_bytes,
    )
    policy = resolve_checkpoint_policy(
        requested_policy,
        local_state_bytes=plan.shards[rank].local_state_bytes,
        device=resolved_device,
    )
    ownership = tuple(
        ParameterGradientOwnership(
            parameter=f"parameter:{index}",
            owner_rank=None,
            occurrence_count=occurrences[index],
            reduction="pending",
            participating_ranks=(),
            distribution="pending",
        )
        for index in range(len(parameters))
    )
    evidence = BackwardExecutionEvidence(
        reduction_counts=[0] * len(parameters), ownership=ownership
    )
    value = _ShardedStatevectorExpectation.apply(
        execution_ir,
        slots,
        policy,
        execution_observable_wires,
        resolved_device,
        process_group,
        evidence,
        *parameters,
    )
    return TorchDistributedStatevectorGradientResult(
        value=value,
        parameters=parameters,
        ownership=ownership,
        checkpoint_policy=policy,
        world_size=world_size,
        local_world_size=plan.local_world_size,
        node_count=plan.node_count,
        backend=str(backend),
        rank=rank,
        local_amplitudes=plan.shards[rank].local_amplitudes,
        total_amplitudes=plan.total_amplitudes,
        logical_to_physical_wires=execution_mapping,
        observable_wires=selected_observable_wires,
        layout_optimization_target="training_step",
        backward_evidence=evidence,
    )


__all__ = (
    "ParameterGradientOwnership",
    "BackwardExecutionEvidence",
    "StatevectorCheckpointPolicy",
    "TorchDistributedStatevectorGradientResult",
    "execute_torch_distributed_statevector_reverse",
    "resolve_checkpoint_policy",
)
