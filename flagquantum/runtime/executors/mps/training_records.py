"""Result records for owner-sharded MPS training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MPSOptimizerOwnership:
    parameter_index: int
    owner_rank: int
    optimizer_state_local: bool
    gradient_route: str
    update_route: str


@dataclass(frozen=True)
class MPSStepMetrics:
    step: int
    loss: float
    forward_seconds: float
    reverse_seconds: float
    optimizer_seconds: float
    end_to_end_seconds: float
    training_compute_seconds: float
    diagnostics_seconds: float
    static_qr_metadata_records: int
    dynamic_metadata_broadcasts: int
    reverse_segment_cache_hit: bool
    gradient_bucket_cache_hit: bool
    layer_halo_message_count: int
    layer_halo_payload_bytes: int
    layer_halo_intra_node_bytes: int
    layer_halo_inter_node_bytes: int
    layer_halo_wait_seconds: float
    owned_site_work: int
    owned_bond_work: int
    boundary_exchanges: int
    svd_activity: int
    qr_activity: int
    peak_memory_bytes: int
    useful_work_completed: bool
    two_site_splits: int = 0
    truncated_splits: int = 0
    discarded_weight: float = 0.0
    boundary_forward_exchanges: int = 0
    boundary_reverse_exchanges: int = 0
    boundary_bytes: int = 0
    bond_updates: tuple[dict[str, Any], ...] = ()
    allocated_memory_bytes: int = 0
    reserved_memory_bytes: int = 0
    tape_memory_bytes: int = 0
    optimizer_memory_bytes: int = 0
    communication_buffer_bytes: int = 0
    gradient_collective_count: int = 0
    gradient_collective_bytes: int = 0
    gradient_bucket_count: int = 0
    gradient_bucket_fill_ratio: float = 0.0
    reverse_tape_segments: int = 0
    fused_reverse_segments: int = 0
    reverse_autograd_grad_invocations: int = 0
    reverse_python_dispatches: int = 0
    qr_factorization_count: int = 0
    svd_factorization_count: int = 0
    objective_scan_pairs: int = 0
    objective_scan_forward_messages: int = 0
    objective_scan_reverse_messages: int = 0
    objective_execution: str = "single_observable_scan"
    learning_rate: float = 0.0
    largest_realized_bond: int = 1
    forward_peak_memory_bytes: int = 0
    reverse_peak_memory_bytes: int = 0
    optimizer_stage: str = "sgd"
    objective_evaluations: int = 1
    parameter_gradients: tuple[tuple[int, float], ...] = ()
    parameter_values: tuple[tuple[int, float], ...] = ()
    optimizer_collective_count: int = 0
    optimizer_collective_bytes: int = 0
    lbfgs_history_length: int = 0


@dataclass(frozen=True)
class ShardedMPSTrainingResult:
    losses: tuple[float, ...]
    completed_steps: int
    start_step: int
    rank: int
    world_size: int
    local_world_size: int
    optimizer: str
    ownership: tuple[MPSOptimizerOwnership, ...]
    steps: tuple[MPSStepMetrics, ...]
    checkpoint_files: tuple[str, ...]
    memory_growth_bytes: int
    suspected_memory_leak: bool
    checkpoint_contract_fingerprint: str = ""
    memory_warmup_steps: int = 0
    communication_setup_seconds: float = 0.0
    site_kernel_execution: str = "eager"
    site_kernel_selection_reason: str = "explicit_eager"
    site_ownership: tuple[tuple[int, ...], ...] = ()
    site_ownership_policy: str = "balanced"
    checkpoint_retention_generations: int | None = 2
    checkpoint_pruned_files: tuple[str, ...] = ()
    checkpoint_write_seconds: float = 0.0
    checkpoint_commit_seconds: float = 0.0
    checkpoint_prune_seconds: float = 0.0
    checkpoint_bytes_written: int = 0
    checkpoint_retained_bytes: int = 0
    checkpoint_free_space_reserve_bytes: int = 1 << 30
    checkpoint_min_free_bytes_observed: int = 0
    checkpoint_estimated_generation_bytes: int = 0
    checkpoint_storage_semantics: str = "local_filesystem"
    checkpoint_storage_probe_seconds: float = 0.0
    checkpoint_overwrite_allowed: bool = False
    checkpoint_deserialization_policy: str = "torch_weights_only"
    checkpoint_writer_lease: str = "exclusive_atomic_file_v1"
    checkpoint_writer_lease_break_allowed: bool = False
    checkpoint_writer_lease_stale_seconds: float = 3600.0
    checkpoint_writer_lease_heartbeat_count: int = 0
    checkpoint_writer_lease_heartbeat_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        sharded = self.world_size > 1
        work = tuple(asdict(item) for item in self.steps)
        idle = not any(item.useful_work_completed for item in self.steps)
        blockers = [
            "release_speedup_evidence_not_attached",
            "release_capacity_evidence_not_attached",
            "multi_gpu_scaling_report_not_attached",
        ]
        if idle:
            blockers.append("rank_has_no_completed_site_or_bond_work")
        if self.suspected_memory_leak:
            blockers.append("multi_step_memory_growth_suspected")
        return {
            "executor": "pytorch_native_sharded_mps_training_v1",
            "distribution_semantics": (
                "sharded_across_ranks" if sharded else "single_device_fast_path"
            ),
            "rank": self.rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": (self.world_size + self.local_world_size - 1)
            // self.local_world_size,
            "optimizer": self.optimizer,
            "completed_steps": self.completed_steps,
            "start_step": self.start_step,
            "losses": self.losses,
            "optimizer_state_ownership_semantics": (
                "sharded_across_ranks" if sharded else "single_device_fast_path"
            ),
            "optimizer_ownership": tuple(asdict(item) for item in self.ownership),
            "step_metrics": work,
            "memory_growth_bytes": self.memory_growth_bytes,
            "suspected_memory_leak": self.suspected_memory_leak,
            "checkpoint_files": self.checkpoint_files,
            "checkpoint_schema": "sharded_mps_training_checkpoint_v2",
            "checkpoint_integrity": "sha256_atomic_file_and_metadata",
            "checkpoint_commit_protocol": "immutable_rank_shards_atomic_manifest_v1",
            "checkpoint_retention_generations": self.checkpoint_retention_generations,
            "checkpoint_pruned_files": self.checkpoint_pruned_files,
            "checkpoint_write_seconds": self.checkpoint_write_seconds,
            "checkpoint_commit_seconds": self.checkpoint_commit_seconds,
            "checkpoint_prune_seconds": self.checkpoint_prune_seconds,
            "checkpoint_bytes_written": self.checkpoint_bytes_written,
            "checkpoint_retained_bytes": self.checkpoint_retained_bytes,
            "checkpoint_free_space_reserve_bytes": (
                self.checkpoint_free_space_reserve_bytes
            ),
            "checkpoint_min_free_bytes_observed": (
                self.checkpoint_min_free_bytes_observed
            ),
            "checkpoint_estimated_generation_bytes": (
                self.checkpoint_estimated_generation_bytes
            ),
            "checkpoint_storage_semantics": self.checkpoint_storage_semantics,
            "checkpoint_storage_probe_seconds": self.checkpoint_storage_probe_seconds,
            "checkpoint_overwrite_allowed": self.checkpoint_overwrite_allowed,
            "checkpoint_deserialization_policy": (
                self.checkpoint_deserialization_policy
            ),
            "checkpoint_writer_lease": self.checkpoint_writer_lease,
            "checkpoint_writer_lease_break_allowed": (
                self.checkpoint_writer_lease_break_allowed
            ),
            "checkpoint_writer_lease_stale_seconds": (
                self.checkpoint_writer_lease_stale_seconds
            ),
            "checkpoint_writer_lease_heartbeat_count": (
                self.checkpoint_writer_lease_heartbeat_count
            ),
            "checkpoint_writer_lease_heartbeat_seconds": (
                self.checkpoint_writer_lease_heartbeat_seconds
            ),
            "checkpoint_contract_fingerprint": self.checkpoint_contract_fingerprint,
            "memory_warmup_steps": self.memory_warmup_steps,
            "communication_setup_seconds": self.communication_setup_seconds,
            "site_kernel_execution": self.site_kernel_execution,
            "site_kernel_selection_reason": self.site_kernel_selection_reason,
            "site_ownership": self.site_ownership,
            "site_ownership_policy": self.site_ownership_policy,
            "communication_protocol": "batched_isend_irecv_packed_multi_tensor_envelope",
            "communication_stream": "dedicated_cuda_stream",
            "rank_useful_work": not idle,
            "full_mps_materialization": False,
            "speedup_gate_passed": False,
            "capacity_gate_passed": False,
            "correctness_gate_passed": self.completed_steps > 0 and not idle,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": tuple(blockers),
        }


__all__ = (
    "MPSOptimizerOwnership",
    "MPSStepMetrics",
    "ShardedMPSTrainingResult",
)
