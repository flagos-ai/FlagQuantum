"""Native PyTorch reverse mode over rank-local statevector shards."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import torch
import torch.distributed as dist

from ....core.ir import CircuitIR, ensure_circuit_ir
from ...builder_compilation import detached_ir_snapshot
from .checkpointing import StatevectorCheckpointPolicy, resolve_checkpoint_policy
from .environment import get_bool
from .forward import (
    communication_aware_wire_layout,
    execute_torch_distributed_statevector,
)
from .kernel_dispatch import (
    KernelDecision,
    KernelDispatchEvidence,
    select_triton_kernel,
)
from .layout import (
    schedule_statevector_dependency_dag,
)
from .state import (
    plan_distributed_statevector,
)

_DEFAULT_REVERSE_CHUNK_AMPLITUDES = 1 << 22


def _reverse_chunk_amplitudes() -> int:
    """Return the bounded reverse exchange chunk configured for this run."""

    raw = os.getenv("FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES")
    value = _DEFAULT_REVERSE_CHUNK_AMPLITUDES if raw is None else int(raw)
    if value <= 0 or value & (value - 1):
        raise ValueError(
            "FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES must be a positive power of two"
        )
    return value


def _triton_vjp_adjoint_decision(*, supported: bool = True) -> KernelDecision:
    requested = os.getenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    return select_triton_kernel("vjp_adjoint", requested=requested, supported=supported)


def _triton_vjp_adjoint_enabled() -> bool:
    return _triton_vjp_adjoint_decision().accelerated


def _communication_aware_layout_enabled() -> bool:
    return os.getenv("FQ_STATEVECTOR_COMM_AWARE_LAYOUT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _reverse_exchange_workspace_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_REVERSE_EXCHANGE_WORKSPACE", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _persistent_wire_layout_enabled() -> bool:
    # Optimized distributed path is the default.  Users can explicitly opt out
    # with FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT=0 for debugging/comparison.
    return get_bool("FQ_SV_PERSISTENT_LAYOUT", True)


def _persistent_inplace_local_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_PERSISTENT_INPLACE_LOCAL", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _fused_vjp_pipeline_enabled() -> bool:
    """Return whether experimental cross-chunk VJP communication overlap is on."""

    return os.getenv("FQ_STATEVECTOR_FUSED_VJP_PIPELINE", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _reverse_cross_shard_cx_packing_enabled() -> bool:
    """Keep reverse CX packing opt-in until backward-only regressions are removed."""

    return os.getenv(
        "FQ_STATEVECTOR_REVERSE_CROSS_SHARD_CX_PACK", "0"
    ).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _communication_aware_wire_layout(
    ir: CircuitIR, *, observable_wire: int, world_size: int
) -> tuple[CircuitIR, int, tuple[int, ...]]:
    """Use the shared alignment-aware layout for differentiable execution."""

    if not _communication_aware_layout_enabled():
        identity = tuple(range(ir.n_wires))
        return ir, observable_wire, identity
    remapped, permutation = communication_aware_wire_layout(
        ir,
        world_size=world_size,
        preferred_local_wires=(observable_wire,),
        optimization_target="training_step",
    )
    return remapped, permutation[observable_wire], permutation


@dataclass
class ParameterGradientOwnership:
    parameter: str
    owner_rank: int | None
    occurrence_count: int
    reduction: str
    participating_ranks: tuple[int, ...]
    distribution: str


@dataclass
class BackwardExecutionEvidence:
    """Evidence populated only by an actually executed autograd backward."""

    status: str = "pending"
    participating_ranks: tuple[int, ...] = ()
    reduction_counts: list[int] = field(default_factory=list)
    error: str | None = None
    ownership: tuple[ParameterGradientOwnership, ...] = ()
    communication_count: int = 0
    communication_bytes: int = 0
    gradient_collective_count: int = 0
    gradient_collective_bytes: int = 0
    persistent_layout_swap_count: int = 0
    persistent_layout_swap_bytes: int = 0
    persistent_layout_buffer_allocation_count: int = 0
    persistent_layout_buffer_reserved_bytes: int = 0
    persistent_inplace_local_enabled: bool = False
    analytic_rotation_derivative_count: int = 0
    fused_parameter_adjoint_count: int = 0
    peak_scratch_bytes: int = 0
    saved_forward_state_reused: bool = False
    exchange_chunk_amplitudes: int = _DEFAULT_REVERSE_CHUNK_AMPLITUDES
    exchange_chunk_bytes: int = 0
    exchange_workspace_allocation_count: int = 0
    exchange_workspace_reuse_count: int = 0
    exchange_workspace_reserved_bytes: int = 0
    exchange_pipeline_prefetch_count: int = 0
    intra_node_communication_count: int = 0
    intra_node_communication_bytes: int = 0
    inter_node_communication_count: int = 0
    inter_node_communication_bytes: int = 0
    kernel_dispatch_evidence: KernelDispatchEvidence = field(
        default_factory=KernelDispatchEvidence
    )


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
            "persistent_layout_enabled": _persistent_wire_layout_enabled(),
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


from .reverse_adjoint import (  # noqa: E402
    _bind_parameters,
    _explicit_sharded_adjoint,
    _local_expectation_z,
    _parameter_layout,
)


def __getattr__(name: str) -> Any:
    """Preserve private compatibility seams moved to ``reverse_adjoint``."""
    module = import_module("flagquantum.runtime.backends.statevector.reverse_adjoint")
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


class _ShardedStatevectorExpectation(torch.autograd.Function):
    """Rematerializing custom-autograd boundary; no global state is saved."""

    @staticmethod
    def forward(
        ctx: Any,
        ir: CircuitIR,
        slots: tuple[tuple[int, str, int], ...],
        policy: StatevectorCheckpointPolicy,
        observable_wire: int,
        device: torch.device,
        process_group: Any | None,
        evidence: BackwardExecutionEvidence,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        ctx.slots = slots
        ctx.policy = policy
        ctx.observable_wire = observable_wire
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
        value = _local_expectation_z(
            result.shard_state,
            plan=result.plan,
            n_wires=ir.n_wires,
            wire=result.logical_to_physical_wires[observable_wire],
        )
        if dist.is_initialized() and dist.get_world_size(process_group) > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        evidence = ctx.evidence
        evidence.status = "running"
        evidence.error = None
        evidence.participating_ranks = ()
        evidence.communication_count = 0
        evidence.communication_bytes = 0
        evidence.gradient_collective_count = 0
        evidence.gradient_collective_bytes = 0
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
                    observable_wire=ctx.observable_wire,
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
    checkpoint_policy: StatevectorCheckpointPolicy | None = None,
    device: torch.device | str | None = None,
    process_group: Any | None = None,
) -> TorchDistributedStatevectorGradientResult:
    """Return a differentiable global Z expectation from rank-local shards."""

    ir = ensure_circuit_ir(circuit_or_ir)
    if observable_wire < 0 or observable_wire >= ir.n_wires:
        raise ValueError(f"observable wire {observable_wire} is outside the circuit")
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
        execution_observable_wire,
        execution_mapping,
    ) = _communication_aware_wire_layout(
        ir, observable_wire=observable_wire, world_size=world_size
    )
    if _persistent_wire_layout_enabled() and world_size > 1:
        execution_ir = schedule_statevector_dependency_dag(
            execution_ir, world_size=world_size
        )
    parameters, slots, occurrences = _parameter_layout(execution_ir)
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
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
        execution_observable_wire,
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
