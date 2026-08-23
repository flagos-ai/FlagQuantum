"""Contracts for deterministic rank-owned MPS reverse execution."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from .canonicalization import MPSCanonicalizationMetrics
from .errors import MPSReverseContractError
from .state import RankOwnedMPSState


@dataclass(frozen=True)
class MPSReverseCheckpointPolicy:
    version: str = "mps_reverse_checkpoint_v3"
    max_saved_bytes: int = 256 * 1024 * 1024
    max_saved_factorization_bytes: int | None = None
    rematerialization_interval: int = 1
    save_two_site_factorizations: bool = False

    def __post_init__(self) -> None:
        if self.max_saved_bytes <= 0 or self.rematerialization_interval <= 0:
            raise ValueError("MPS reverse checkpoint bounds must be positive")
        if (
            self.max_saved_factorization_bytes is not None
            and self.max_saved_factorization_bytes < 0
        ):
            raise ValueError("MPS factorization checkpoint bound must be non-negative")
        if self.rematerialization_interval != 1:
            raise ValueError(
                "distributed MPS rematerialization intervals greater than one "
                "are not implemented; use interval=1"
            )


@dataclass
class ReverseCheckpointBudget:
    """Track rank-local reservations against collective memory limits."""

    policy: MPSReverseCheckpointPolicy
    device: torch.device
    saved_bytes: int = 0
    saved_factorization_bytes: int = 0
    def _collective_max(self, candidate: int) -> int:
        maximum = torch.tensor([candidate], dtype=torch.int64, device=self.device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        return int(maximum.item())

    def reserve(self, byte_count: int) -> None:
        candidate = self.saved_bytes + int(byte_count)
        collective_maximum = self._collective_max(candidate)
        if collective_maximum > self.policy.max_saved_bytes:
            raise MPSReverseContractError(
                f"MPS reverse checkpoint would require {collective_maximum} bytes, "
                f"limit is {self.policy.max_saved_bytes}"
            )
        self.saved_bytes = candidate

    def reserve_factorization(self, byte_count: int) -> bool:
        """Reserve an optional factorization without exceeding its pool."""

        if self.policy.max_saved_factorization_bytes is None:
            self.reserve(byte_count)
            return True
        candidate = self.saved_factorization_bytes + int(byte_count)
        if self._collective_max(candidate) > self.policy.max_saved_factorization_bytes:
            return False
        self.saved_factorization_bytes = candidate
        self.saved_bytes += int(byte_count)
        return True

    def reserve_factorization_batch(self, byte_counts: Sequence[int]) -> bool:
        """Reserve a compiled layer with one collective when the layer fits."""

        total = sum(int(value) for value in byte_counts)
        if self.policy.max_saved_factorization_bytes is None:
            self.reserve(total)
            return True
        candidate = self.saved_factorization_bytes + total
        if self._collective_max(candidate) > self.policy.max_saved_factorization_bytes:
            return False
        self.saved_factorization_bytes = candidate
        self.saved_bytes += total
        return True


@dataclass(frozen=True)
class MPSReverseTapeRecord:
    operation_id: str
    forward_sequence: int
    reverse_sequence: int
    kind: str
    wires: tuple[int, ...]
    compute_owner: int
    owner_ranks: tuple[int, ...]
    communication_peer: int | None
    communication_sequence: int | None
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    parameter_indices: tuple[int, ...]
    discarded_weight: float
    kept_rank: int | None
    original_rank: int | None
    singular_value_gap: float | None
    rematerialize: bool
    factorization_method: str | None = None

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MPSReverseTape:
    records: tuple[MPSReverseTapeRecord, ...]
    identity: str
    saved_tensor_bytes: int
    checkpoint_policy: MPSReverseCheckpointPolicy

    @classmethod
    def build(
        cls,
        records: list[MPSReverseTapeRecord],
        *,
        saved_tensor_bytes: int,
        checkpoint_policy: MPSReverseCheckpointPolicy,
    ) -> "MPSReverseTape":
        normalized = tuple(
            MPSReverseTapeRecord(
                **{
                    **record.summary(),
                    "reverse_sequence": len(records) - index - 1,
                }
            )
            for index, record in enumerate(records)
        )
        content = json.dumps(
            [record.summary() for record in normalized],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return cls(
            records=normalized,
            identity=hashlib.sha256(content).hexdigest(),
            saved_tensor_bytes=int(saved_tensor_bytes),
            checkpoint_policy=checkpoint_policy,
        )

    def validate_distributed(self, *, identity_override: str | None = None) -> None:
        local = identity_override or self.identity
        sequences = tuple(record.reverse_sequence for record in reversed(self.records))
        if sequences != tuple(range(len(self.records))):
            raise MPSReverseContractError("MPS reverse sequence is not contiguous")
        if dist.is_initialized():
            digest = hashlib.sha256(str(local).encode()).digest()
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if dist.get_backend() == "nccl"
                else torch.device("cpu")
            )
            encoded = torch.tensor(tuple(digest), dtype=torch.uint8, device=device)
            gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, encoded)
            identities = tuple(bytes(value.cpu().tolist()).hex() for value in gathered)
            if len(set(identities)) != 1:
                raise MPSReverseContractError(
                    f"distributed MPS reverse tape mismatch: {identities}"
                )


@dataclass(frozen=True)
class MPSParameterGradientOwnership:
    parameter_index: int
    owner_ranks: tuple[int, ...]
    occurrence_count: int
    reduction: str


@dataclass
class TorchDistributedMPSGradientResult:
    value: torch.Tensor
    parameters: tuple[torch.Tensor, ...]
    tape: MPSReverseTape
    ownership: tuple[MPSParameterGradientOwnership, ...]
    discarded_weight: float
    gradient_policy: str
    gradient_tolerance: float
    world_size: int
    rank: int
    _backward: Any
    objective_scan_pairs: int = 0
    objective_scan_forward_messages: int = 0
    objective_scan_reverse_messages: int = 0
    objective_execution: str = "single_observable_scan"
    reverse_tape_segments: int = 0
    fused_reverse_segments: int = 0
    reverse_autograd_grad_invocations: int = 0
    reverse_python_dispatches: int = 0
    gradient_collective_count: int = 0
    gradient_collective_bytes: int = 0
    gradient_bucket_count: int = 0
    gradient_bucket_fill_ratio: float = 0.0
    gradient_owner_ranks: tuple[int, ...] = ()
    canonicalization_policy: str = "dirty"
    initial_mps_canonical: bool = False
    dirty_bonds: tuple[int, ...] = ()
    planned_canonicalization_bonds: tuple[int, ...] = ()
    qr_factorization_count: int = 0
    svd_factorization_count: int = 0
    static_qr_metadata_records: int = 0
    dynamic_metadata_broadcasts: int = 0
    reverse_segment_cache_hit: bool = False
    gradient_bucket_cache_hit: bool = False
    layer_halo_message_count: int = 0
    layer_halo_payload_bytes: int = 0
    layer_halo_intra_node_bytes: int = 0
    layer_halo_inter_node_bytes: int = 0
    layer_halo_wait_seconds: float = 0.0
    _backward_status: str = "ready"
    _backward_error: str | None = None
    _last_completed_record: int | None = None

    def backward(self) -> None:
        if self._backward_status != "ready":
            raise MPSReverseContractError(
                f"MPS reverse result cannot run backward from "
                f"{self._backward_status!r} state"
            )
        self._backward_status = "running"
        try:
            self._backward(self)
        except Exception as error:
            self._backward_status = "failed"
            self._backward_error = f"{type(error).__name__}: {error}"
            raise
        self._backward_status = "completed"

    def full_state(self) -> torch.Tensor:
        raise MPSReverseContractError("MPS reverse forbids full-state materialization")

    def diagnostics(self) -> dict[str, Any]:
        cursor = (
            0
            if self._last_completed_record is None
            else self._last_completed_record + 1
        )
        active = next(
            (
                record
                for record in reversed(self.tape.records)
                if record.reverse_sequence == cursor
            ),
            None,
        )
        return {
            "rank": self.rank,
            "tape_cursor": cursor,
            "reverse_sequence": active.reverse_sequence if active else None,
            "operation_id": active.operation_id if active else None,
            "bond": min(active.wires) if active and len(active.wires) == 2 else None,
            "communication_peer": active.communication_peer if active else None,
            "last_completed_record": self._last_completed_record,
        }

    def summary(self) -> dict[str, Any]:
        approximate = self.discarded_weight > 0
        return {
            "executor": "pytorch_explicit_rank_owned_mps_reverse_v1",
            "world_size": self.world_size,
            "rank": self.rank,
            "distribution_semantics": "sharded_across_ranks",
            "mps_backward_execution": self._backward_status,
            "backward_error": self._backward_error,
            "reverse_tape_identity": self.tape.identity,
            "reverse_tape_records": len(self.tape.records),
            "saved_tensor_bytes": self.tape.saved_tensor_bytes,
            "checkpoint_policy": asdict(self.tape.checkpoint_policy),
            "last_completed_record": self._last_completed_record,
            "watchdog_diagnostics": self.diagnostics(),
            "parameter_ownership": tuple(asdict(value) for value in self.ownership),
            "shared_parameter_reduction": (
                "bounded_dtype_owner_reduce_then_broadcast"
                if self.gradient_owner_ranks
                else "bounded_dtype_all_reduce_sum"
            ),
            "gradient_policy": self.gradient_policy,
            "gradient_accuracy": "approximate" if approximate else "exact",
            "gradient_tolerance": self.gradient_tolerance,
            "discarded_weight": self.discarded_weight,
            "objective_scan_pairs": self.objective_scan_pairs,
            "objective_scan_forward_messages": self.objective_scan_forward_messages,
            "objective_scan_reverse_messages": self.objective_scan_reverse_messages,
            "objective_execution": self.objective_execution,
            "reverse_tape_segments": self.reverse_tape_segments,
            "fused_reverse_segments": self.fused_reverse_segments,
            "reverse_autograd_grad_invocations": self.reverse_autograd_grad_invocations,
            "reverse_python_dispatches": self.reverse_python_dispatches,
            "gradient_collective_count": self.gradient_collective_count,
            "gradient_collective_bytes": self.gradient_collective_bytes,
            "gradient_bucket_count": self.gradient_bucket_count,
            "gradient_bucket_fill_ratio": self.gradient_bucket_fill_ratio,
            "gradient_owner_ranks": self.gradient_owner_ranks,
            "canonicalization_policy": self.canonicalization_policy,
            "initial_mps_canonical": self.initial_mps_canonical,
            "dirty_bonds": self.dirty_bonds,
            "planned_canonicalization_bonds": self.planned_canonicalization_bonds,
            "qr_factorization_count": self.qr_factorization_count,
            "svd_factorization_count": self.svd_factorization_count,
            "static_qr_metadata_records": self.static_qr_metadata_records,
            "dynamic_metadata_broadcasts": self.dynamic_metadata_broadcasts,
            "reverse_segment_cache_hit": self.reverse_segment_cache_hit,
            "gradient_bucket_cache_hit": self.gradient_bucket_cache_hit,
            "layer_halo_message_count": self.layer_halo_message_count,
            "layer_halo_payload_bytes": self.layer_halo_payload_bytes,
            "layer_halo_intra_node_bytes": self.layer_halo_intra_node_bytes,
            "layer_halo_inter_node_bytes": self.layer_halo_inter_node_bytes,
            "layer_halo_wait_seconds": self.layer_halo_wait_seconds,
            "canonicalization_pullback": "explicit_owner_local_qr_vjp",
            "truncation_pullback": "explicit_owner_local_svd_vjp",
            "singular_value_degeneracy_policy": "fail_closed_at_truncation_boundary",
            "replicated_autograd": False,
            "statevector_fallback": False,
            "full_mps_reconstruction": False,
            "jax_required": False,
            "blockers": (),
        }


@dataclass(frozen=True)
class TorchDistributedMPSForwardResult:
    """Rank-owned forward state with stable execution-summary metadata."""

    shard_state: RankOwnedMPSState
    local_gate_count: int
    boundary_gate_count: int
    boundary_messages: int
    boundary_bytes: int
    rebalance_messages: int
    rebalance_bytes: int
    rebalance_count: int
    partition_history: tuple[tuple[tuple[int, ...], ...], ...]
    bond_dimensions: tuple[int, ...]
    rank_tensor_bytes: tuple[int, ...]
    canonicalization: MPSCanonicalizationMetrics
    truncation_records: tuple[dict[str, Any], ...]
    global_error_budget: float | None
    error_budget_policy: str
    truncation_gradient_policy: str
    backend: str
    layer_lifecycle_records: tuple[dict[str, Any], ...] = ()
    factorization_records: tuple[dict[str, Any], ...] = ()
    layer_cache_empty_at_return: bool = True
    gate_matrix_materialization: str = "per_instruction_last_use"

    def full_state(self) -> torch.Tensor:
        return self.shard_state.full_state()

    def summary(self) -> dict[str, Any]:
        local_world_size = min(
            self.shard_state.world_size,
            int(os.environ.get("LOCAL_WORLD_SIZE", self.shard_state.world_size)),
        )
        truncation_error = sum(
            float(record["discarded_weight"]) for record in self.truncation_records
        )
        approximate = truncation_error > 0.0
        return {
            "executor": "pytorch_native_rank_owned_mps_forward_v1",
            "state_mode": "distributed_mps",
            "distribution_semantics": (
                "sharded_across_ranks"
                if self.shard_state.world_size > 1
                else "single_device_fast_path"
            ),
            "backend": self.backend,
            "claim_evidence_type": (
                "accelerator_semantics"
                if self.backend == "nccl"
                else (
                    "local_semantics"
                    if self.shard_state.world_size == 1
                    else "development_semantics"
                )
            ),
            "world_size": self.shard_state.world_size,
            "local_world_size": local_world_size,
            "node_count": (self.shard_state.world_size + local_world_size - 1)
            // local_world_size,
            "rank": self.shard_state.rank,
            "rank_placement": {
                "rank": self.shard_state.rank,
                "local_rank": int(
                    os.environ.get(
                        "LOCAL_RANK", self.shard_state.rank % local_world_size
                    )
                ),
                "node_rank": self.shard_state.rank // local_world_size,
            },
            "rank_ownership": self.shard_state.summary(),
            "local_gate_count": self.local_gate_count,
            "boundary_gate_count": self.boundary_gate_count,
            "boundary_messages": self.boundary_messages,
            "boundary_bytes": self.boundary_bytes,
            "rebalance_messages": self.rebalance_messages,
            "rebalance_bytes": self.rebalance_bytes,
            "communication_messages": (
                self.boundary_messages
                + self.rebalance_messages
                + self.canonicalization.messages_by_rank[self.shard_state.rank]
            ),
            "communication_bytes": (
                self.boundary_bytes
                + self.rebalance_bytes
                + self.canonicalization.bytes_by_rank[self.shard_state.rank]
            ),
            "communication_primitive": "torch.distributed_point_to_point",
            "local_tensor_bytes_by_rank": self.rank_tensor_bytes,
            "rebalance_count": self.rebalance_count,
            "partition_history": self.partition_history,
            "bond_dimensions": self.bond_dimensions,
            "orthogonality_center": self.canonicalization.center,
            "mixed_canonical_residual": self.canonicalization.residual,
            "state_norms": self.canonicalization.state_norms,
            "canonicalization_messages_by_rank": self.canonicalization.messages_by_rank,
            "canonicalization_bytes_by_rank": self.canonicalization.bytes_by_rank,
            "canonicalization_temporary_bytes_by_rank": self.canonicalization.temporary_bytes_by_rank,
            "truncation_records": self.truncation_records,
            "truncation_error": truncation_error,
            "truncation_semantics": "approximate" if approximate else "exact",
            "global_error_budget": self.global_error_budget,
            "error_budget_policy": self.error_budget_policy,
            "error_budget_satisfied": (
                None
                if self.global_error_budget is None
                else truncation_error <= self.global_error_budget
            ),
            "truncation_gradient_metadata": {
                "policy": self.truncation_gradient_policy,
                "status": (
                    "unsupported"
                    if self.truncation_gradient_policy == "unsupported"
                    else self.truncation_gradient_policy
                ),
                "exact": (
                    self.truncation_gradient_policy == "exact" and not approximate
                ),
            },
            "forward_tensor_lifetime": {
                "policy": "layer_local_destructive_consumption_v1",
                "gate_matrix_materialization": self.gate_matrix_materialization,
                "layer_cache_empty_at_return": self.layer_cache_empty_at_return,
                "layer_drain_count": len(self.layer_lifecycle_records),
                "layer_records": self.layer_lifecycle_records,
            },
            "factorization_workspace": {
                "policy": "workspace_aware_microbatch_v1",
                "decision_count": len(self.factorization_records),
                "records": self.factorization_records,
            },
            "full_mps_reconstruction_count": 0,
            "full_state_materialization": False,
            "jax_required": False,
            "jax_imported": any(
                name == "jax" or name.startswith("jax.") for name in sys.modules
            ),
            "scalability_claim_allowed": False,
            "blockers": (
                "mps_sharded_backward_pending",
                "mps_sharded_optimizer_pending",
                "accelerator_capacity_acceptance_pending",
            ),
        }


def build_mps_reverse_tape_record(
    *,
    index: int,
    kind: str,
    wires: tuple[int, ...],
    compute_owner: int,
    owner_ranks: tuple[int, ...],
    input_shapes: Sequence[Sequence[int]],
    output_shapes: Sequence[Sequence[int]],
    parameter_indices: tuple[int, ...] = (),
    split_info: Mapping[str, Any] | None = None,
) -> MPSReverseTapeRecord:
    """Build one deterministic reverse-tape record and operation identity."""
    content = f"{index}:{kind}:{wires}:{compute_owner}".encode()
    peer = next((rank for rank in owner_ranks if rank != compute_owner), None)
    info = dict(split_info or {})
    return MPSReverseTapeRecord(
        operation_id=hashlib.sha256(content).hexdigest()[:16],
        forward_sequence=index,
        reverse_sequence=0,
        kind=kind,
        wires=wires,
        compute_owner=compute_owner,
        owner_ranks=owner_ranks,
        communication_peer=peer,
        communication_sequence=index if peer is not None else None,
        input_shapes=tuple(tuple(int(v) for v in shape) for shape in input_shapes),
        output_shapes=tuple(tuple(int(v) for v in shape) for shape in output_shapes),
        parameter_indices=parameter_indices,
        discarded_weight=float(info.get("discarded_weight", 0.0)),
        kept_rank=int(info["rank"]) if "rank" in info else None,
        original_rank=int(info["original_rank"]) if "original_rank" in info else None,
        singular_value_gap=(
            float(info["singular_value_gap"]) if "singular_value_gap" in info else None
        ),
        rematerialize=True,
        factorization_method=info.get("method"),
    )


__all__ = [
    "MPSParameterGradientOwnership",
    "MPSReverseCheckpointPolicy",
    "MPSReverseContractError",
    "MPSReverseTape",
    "MPSReverseTapeRecord",
    "ReverseCheckpointBudget",
    "TorchDistributedMPSGradientResult",
    "TorchDistributedMPSForwardResult",
    "build_mps_reverse_tape_record",
]
