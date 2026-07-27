"""Distributed statevector contracts and immutable records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...distributed.backend_policy import (
    DistributedBackendPolicy,
)

_DIAGONAL_GATES = {
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rz",
    "phase",
    "p",
    "u1",
    "cz",
    "cphase",
}
_TARGET_LAST_GATES = {"cx", "cnot", "cy", "crx", "cry", "crz"}
_PARAM_ALIASES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
    "crx": ("theta",),
    "cry": ("theta",),
    "crz": ("theta",),
    "cphase": ("theta",),
    "rxx": ("theta",),
    "ryy": ("theta",),
    "rzz": ("theta",),
}


@dataclass(frozen=True)
class StatevectorShard:
    """Contiguous amplitude range owned by one distributed rank."""

    rank: int
    world_size: int
    amplitude_start: int
    amplitude_end: int
    local_amplitudes: int
    local_state_bytes: int


@dataclass(frozen=True)
class StatevectorShardState:
    """Rank-local dense statevector shard used by numerical sharded executors."""

    rank: int
    shard: StatevectorShard
    amplitudes: Any
    global_indices: Any

    def summary(self) -> dict[str, Any]:
        compact = int(self.global_indices.numel()) == 0
        return {
            "rank": self.rank,
            "amplitude_start": self.shard.amplitude_start,
            "amplitude_end": self.shard.amplitude_end,
            "local_amplitudes": self.shard.local_amplitudes,
            "global_index_count": self.shard.local_amplitudes,
            "compact_global_indices": compact,
            "shape": tuple(self.amplitudes.shape),
            "device": str(self.amplitudes.device),
            "dtype": str(self.amplitudes.dtype),
        }


@dataclass(frozen=True)
class LocalDistributedStatevectorResult:
    """Single-process CPU simulation of a distributed statevector execution."""

    state: Any
    shards: tuple[StatevectorShardState, ...]
    plan: Any
    local_gate_count: int
    distributed_gate_count: int
    simulated_communication_count: int
    simulated_communication_bytes: int
    full_state_reconstruction_count: int
    backend_policy: DistributedBackendPolicy
    jax_distributed_plan: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "executor": "local_cpu_distributed_simulator",
            "claim_evidence_type": "development_smoke",
            "distributed_backend_policy": self.backend_policy.summary(),
            "distribution_semantics": self.plan.summary()["distribution_semantics"],
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": bool(
                self.plan.world_size > 1
                and self.plan.distribution != "replicated_single_rank"
            ),
            "scalability_note": (
                "Single-process CPU simulator for distributed correctness. It uses the same shard "
                "ownership plan but is not a production communication benchmark."
            ),
            "world_size": self.plan.world_size,
            "local_world_size": self.plan.local_world_size,
            "node_count": self.plan.node_count,
            "n_wires": self.plan.n_wires,
            "batch_size": self.plan.bsz,
            "local_gate_count": self.local_gate_count,
            "distributed_gate_count": self.distributed_gate_count,
            "simulated_communication_count": self.simulated_communication_count,
            "simulated_communication_bytes": self.simulated_communication_bytes,
            "communication_execution": "rank_local_amplitude_exchange",
            "full_state_reconstruction_count": self.full_state_reconstruction_count,
            "jax_distributed_plan": self.jax_distributed_plan,
            "rank_shards": tuple(shard.summary() for shard in self.shards),
        }


@dataclass(frozen=True)
class StatevectorCommunicationEdge:
    """Logical communication edge between two ranks."""

    src_rank: int
    dst_rank: int
    communication: str
    segment_indices: tuple[int, ...]
    estimated_transfer_bytes: int
    tier: str = "unknown"


@dataclass(frozen=True)
class StatevectorRankTopology:
    """Rank layout and logical communication graph for dense-state execution."""

    world_size: int
    local_world_size: int
    node_count: int
    layout: str
    rank_coordinates: tuple[tuple[int, ...], ...]
    edges: tuple[StatevectorCommunicationEdge, ...]

    def summary(self) -> dict[str, Any]:
        intra_bytes = sum(
            edge.estimated_transfer_bytes
            for edge in self.edges
            if edge.tier == "intra_node"
        )
        inter_bytes = sum(
            edge.estimated_transfer_bytes
            for edge in self.edges
            if edge.tier == "inter_node"
        )
        return {
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "layout": self.layout,
            "rank_coordinates": self.rank_coordinates,
            "edge_count": len(self.edges),
            "communications": tuple(
                sorted({edge.communication for edge in self.edges})
            ),
            "communication_tiers": tuple(sorted({edge.tier for edge in self.edges})),
            "intra_node_communication_bytes": intra_bytes,
            "inter_node_communication_bytes": inter_bytes,
        }


@dataclass(frozen=True)
class StatevectorGatePlan:
    """Communication-aware plan for one statevector gate."""

    index: int
    name: str
    wires: tuple[int, ...]
    layer: int
    execution: str
    communication: str
    sharded_wires_touched: tuple[int, ...]
    estimated_transfer_bytes: int = 0

    @property
    def requires_communication(self) -> bool:
        return self.communication != "local"


@dataclass(frozen=True)
class StatevectorFusionBlock:
    """A contiguous block of gates that can be fused before a communication barrier."""

    index: int
    gate_indices: tuple[int, ...]
    wires: tuple[int, ...]
    communication_barrier: bool
    estimated_gate_width: int


@dataclass(frozen=True)
class StatevectorExecutionSegment:
    """A schedulable local-fusion or communication batch."""

    index: int
    kind: str
    gate_indices: tuple[int, ...]
    communication: str
    wires: tuple[int, ...]
    estimated_transfer_bytes: int
    can_overlap_with_compute: bool = False


@dataclass(frozen=True)
class StatevectorPerformanceEstimate:
    """Dry-run performance estimate for a distributed statevector plan."""

    total_estimated_seconds: float
    compute_seconds: float
    communication_seconds: float
    overlapped_seconds: float
    bottleneck: str
    bandwidth_bytes_per_second: float
    local_gate_amplitudes_per_second: float

    def summary(self) -> dict[str, Any]:
        return {
            "total_estimated_seconds": self.total_estimated_seconds,
            "compute_seconds": self.compute_seconds,
            "communication_seconds": self.communication_seconds,
            "overlapped_seconds": self.overlapped_seconds,
            "bottleneck": self.bottleneck,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "local_gate_amplitudes_per_second": self.local_gate_amplitudes_per_second,
        }


@dataclass(frozen=True)
class StatevectorTraceEvent:
    """One rank-local dry-run event for a distributed statevector segment."""

    rank: int
    step: int
    segment_index: int
    action: str
    communication: str
    gate_indices: tuple[int, ...]
    peer_ranks: tuple[int, ...]
    buffer_bytes: int = 0


@dataclass(frozen=True)
class StatevectorTraceReport:
    """Dry-run trace and consistency report for a distributed statevector plan."""

    valid: bool
    errors: tuple[str, ...]
    events: tuple[StatevectorTraceEvent, ...]
    per_rank_event_counts: tuple[int, ...]
    peak_rank_buffer_bytes: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "event_count": len(self.events),
            "per_rank_event_counts": self.per_rank_event_counts,
            "peak_rank_buffer_bytes": self.peak_rank_buffer_bytes,
        }


@dataclass(frozen=True)
class StatevectorSegmentResult:
    """Dry-run executor result for one segment on one rank."""

    rank: int
    segment_index: int
    action: str
    gate_indices: tuple[int, ...]
    local_bytes_processed: int
    communication_bytes: int
    buffer_bytes: int
    peer_ranks: tuple[int, ...]


@dataclass(frozen=True)
class StatevectorRankResult:
    """Aggregated dry-run executor result for one rank."""

    rank: int
    segment_results: tuple[StatevectorSegmentResult, ...]
    local_bytes_processed: int
    communication_bytes: int
    peak_buffer_bytes: int


@dataclass(frozen=True)
class StatevectorExecutorReport:
    """Rank-complete dry-run executor report for a distributed statevector plan."""

    valid: bool
    errors: tuple[str, ...]
    rank_results: tuple[StatevectorRankResult, ...]
    total_local_bytes_processed: int
    total_communication_bytes: int
    peak_buffer_bytes: int
    trace: StatevectorTraceReport

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "rank_count": len(self.rank_results),
            "total_local_bytes_processed": self.total_local_bytes_processed,
            "total_communication_bytes": self.total_communication_bytes,
            "peak_buffer_bytes": self.peak_buffer_bytes,
            "trace_event_count": len(self.trace.events),
        }


@dataclass(frozen=True)
class StatevectorCorrectnessRunSpec:
    """Launch specification for real multi-rank statevector correctness checks."""

    world_size: int
    backend: str
    entrypoint: str
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    expected_checks: tuple[str, ...]

    def command(self) -> tuple[str, ...]:
        return (
            "torchrun",
            f"--nproc_per_node={self.world_size}",
            self.entrypoint,
            *self.args,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "backend": self.backend,
            "entrypoint": self.entrypoint,
            "command": self.command(),
            "expected_checks": self.expected_checks,
        }


@dataclass(frozen=True)
class StatevectorTransportEvent:
    """One real torch.distributed transport event for a statevector segment."""

    rank: int
    segment_index: int
    communication: str
    peer_ranks: tuple[int, ...]
    sent_values: tuple[int, ...]
    received_values: tuple[int, ...]


@dataclass(frozen=True)
class StatevectorTransportReport:
    """Report from the real distributed segment transport smoke executor."""

    rank: int
    world_size: int
    valid: bool
    errors: tuple[str, ...]
    events: tuple[StatevectorTransportEvent, ...]
    pair_exchange_count: int
    all_to_all_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "valid": self.valid,
            "error_count": len(self.errors),
            "event_count": len(self.events),
            "pair_exchange_count": self.pair_exchange_count,
            "all_to_all_count": self.all_to_all_count,
        }


@dataclass(frozen=True)
class StatevectorBufferPlan:
    """Reusable communication buffer budget for distributed statevector execution."""

    segment_index: int
    communication: str
    send_buffer_bytes: int
    recv_buffer_bytes: int
    scratch_buffer_bytes: int
    reusable: bool = True

    @property
    def peak_bytes(self) -> int:
        return (
            self.send_buffer_bytes + self.recv_buffer_bytes + self.scratch_buffer_bytes
        )


@dataclass(frozen=True)
class DistributedStatevectorPlan:
    """Full dense-state distributed execution plan."""

    n_wires: int
    bsz: int
    world_size: int
    local_world_size: int
    node_count: int
    complex_bytes: int
    total_amplitudes: int
    total_state_bytes: int
    per_rank_state_bytes: int
    rank_address_bits: int
    sharded_wires: tuple[int, ...]
    shards: tuple[StatevectorShard, ...]
    gate_plans: tuple[StatevectorGatePlan, ...]
    fusion_blocks: tuple[StatevectorFusionBlock, ...]
    execution_segments: tuple[StatevectorExecutionSegment, ...]
    topology: StatevectorRankTopology
    buffer_plans: tuple[StatevectorBufferPlan, ...]
    layers: tuple[tuple[int, ...], ...]
    communication_gate_count: int
    estimated_transfer_bytes: int
    distribution: str

    def summary(self) -> dict[str, Any]:
        is_sharded = (
            self.world_size > 1 and self.distribution != "replicated_single_rank"
        )
        intra_bytes = sum(
            edge.estimated_transfer_bytes
            for edge in self.topology.edges
            if edge.tier == "intra_node"
        )
        inter_bytes = sum(
            edge.estimated_transfer_bytes
            for edge in self.topology.edges
            if edge.tier == "inter_node"
        )
        rank_shards = tuple(
            {
                "rank": shard.rank,
                "amplitude_start": shard.amplitude_start,
                "amplitude_end": shard.amplitude_end,
                "local_amplitudes": shard.local_amplitudes,
                "local_state_bytes": shard.local_state_bytes,
            }
            for shard in self.shards
        )
        topology_summary = self.topology.summary()
        memory_plan = {
            "state_partition": "amplitude_or_qubit_address_shards",
            "per_rank_shard_bytes": tuple(
                int(shard.local_state_bytes) for shard in self.shards
            ),
            "communication_buffer_bytes": max(
                (buffer_plan.peak_bytes for buffer_plan in self.buffer_plans),
                default=0,
            ),
            "communication_buffer_count": len(self.buffer_plans),
            "requires_per_rank_memory_evidence": True,
        }
        communication_plan = {
            "communication_execution": "planned_statevector_pair_exchange_or_all_to_all",
            "transport_patterns": tuple(topology_summary.get("communications", ())),
            "collective_blockers": ("production_backward_collective_runtime_pending",),
            "topology_dependency": "collective_route_depends_on_runtime_backend",
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "intra_node_communication_bytes": intra_bytes,
            "inter_node_communication_bytes": inter_bytes,
        }
        payload = {
            "state_mode": "distributed_statevector",
            "claim_evidence_type": "plan_preflight",
            "distribution": self.distribution,
            "distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "forward_distribution_semantics": (
                "sharded_across_ranks" if is_sharded else "replicated_single_rank"
            ),
            "backward_distribution_semantics": "incomplete",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sharding_plan_available": bool(is_sharded),
            "parameter_gradient_ready": False,
            "optimizer_update_semantics": "not_measured",
            "backward_uses_full_state_replay": False,
            "n_wires": self.n_wires,
            "bsz": self.bsz,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "rank_address_bits": self.rank_address_bits,
            "sharded_wires": self.sharded_wires,
            "total_state_bytes": self.total_state_bytes,
            "per_rank_state_bytes": self.per_rank_state_bytes,
            "rank_shards": rank_shards,
            "local_memory_bytes_by_rank": tuple(
                int(shard.local_state_bytes) for shard in self.shards
            ),
            "memory_plan": memory_plan,
            "communication_gate_count": self.communication_gate_count,
            "estimated_transfer_bytes": self.estimated_transfer_bytes,
            "intra_node_communication_bytes": intra_bytes,
            "inter_node_communication_bytes": inter_bytes,
            "communication_plan": communication_plan,
            "communication_tiers": topology_summary,
            "communication_tier_model": "rank_endpoint_attributed",
            "fusion_block_count": len(self.fusion_blocks),
            "execution_segment_count": len(self.execution_segments),
            "topology_layout": self.topology.layout,
            "topology_edge_count": len(self.topology.edges),
            "peak_buffer_bytes": max(
                (buffer_plan.peak_bytes for buffer_plan in self.buffer_plans),
                default=0,
            ),
            "communication_buffer_count": len(self.buffer_plans),
            "communication_segment_count": sum(
                1
                for segment in self.execution_segments
                if segment.communication != "local"
            ),
            "overlap_candidate_count": sum(
                1
                for segment in self.execution_segments
                if segment.can_overlap_with_compute
            ),
            "gate_count": len(self.gate_plans),
        }
        from ...audit import evaluate_statevector_training_claimability

        gate = evaluate_statevector_training_claimability(payload).summary()
        payload["statevector_training_claimability_gate"] = gate
        payload["statevector_training_claimability_status"] = gate["status"]
        # Compatibility keys for historical benchmark payloads.
        payload["phase4_claimability_gate"] = gate
        payload["phase4_claimability_status"] = gate["status"]
        payload["claimable_production_training"] = gate["claimable_production_training"]
        return payload

    def estimate_performance(
        self,
        *,
        bandwidth_bytes_per_second: float = 50e9,
        local_gate_amplitudes_per_second: float = 5e11,
    ) -> StatevectorPerformanceEstimate:
        from .planning import estimate_distributed_statevector_performance

        return estimate_distributed_statevector_performance(
            self,
            bandwidth_bytes_per_second=bandwidth_bytes_per_second,
            local_gate_amplitudes_per_second=local_gate_amplitudes_per_second,
        )

    def trace(self) -> StatevectorTraceReport:
        from .planning import trace_distributed_statevector_plan

        return trace_distributed_statevector_plan(self)

    def validate(self) -> StatevectorTraceReport:
        from .planning import validate_distributed_statevector_plan

        return validate_distributed_statevector_plan(self)

    def execute_dry_run(self) -> StatevectorExecutorReport:
        from .local_execution import execute_distributed_statevector_dry_run

        return execute_distributed_statevector_dry_run(self)

    def correctness_run_spec(
        self,
        *,
        entrypoint: str = "tests/distributed/statevector_correctness.py",
        backend: str = "gloo",
    ) -> StatevectorCorrectnessRunSpec:
        from .local_execution import build_statevector_correctness_run_spec

        return build_statevector_correctness_run_spec(
            self,
            entrypoint=entrypoint,
            backend=backend,
        )
