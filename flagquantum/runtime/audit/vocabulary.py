"""Canonical vocabulary for distributed audit payloads."""

SCALABLE_DISTRIBUTION_SEMANTICS = frozenset({"sharded_across_ranks"})
RELEASE_CLAIM_EVIDENCE_TYPES = frozenset(
    {"production_training_benchmark", "release_payload"}
)
STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES = frozenset(
    {"claimable_production_training", "preflight_only", "local_simulation", "blocked"}
)
MPS_BACKWARD_READINESS_STATUSES = frozenset(
    {
        "blocked",
        "control_plane_ready",
        "backward_preflight_ready",
        "production_backward_evidence",
        "production_training_claimable",
    }
)
CLAIMABILITY_STATUSES = STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES
TRANSPORT_EVIDENCE_STATUSES = frozenset(
    {
        "topology_dependent_planning",
        "single_node_executed_collective",
        "multi_node_production_transport",
        "blocked",
    }
)
MULTI_NODE_TRANSPORT_BACKENDS = frozenset(
    {
        "nccl",
        "gloo",
        "xla",
        "xla_collective",
        "xla_multi_node_collective",
        "torch_distributed",
        "mpi",
    }
)
STATEVECTOR_STATE_MODES = frozenset(
    {"distributed_statevector", "jax_sharded_statevector", "statevector"}
)
MPS_STATE_MODES = frozenset({"distributed_mps", "jax_sharded_mps", "mps"})
EVIDENCE_BACKEND_FAMILIES = frozenset(
    {"statevector", "mps", "tensor_network", "unknown"}
)
CLAIM_EVIDENCE_TYPES = frozenset(
    {
        "plan_preflight",
        "development_smoke",
        "production_runtime",
        "production_training_benchmark",
        "release_payload",
        "unknown",
    }
)
SINGLE_DEVICE_DISTRIBUTION_SEMANTICS = frozenset({"single_device_fast_path"})
REPLICATED_DISTRIBUTION_SEMANTICS = frozenset(
    {
        "replicated_single_rank",
        "replicated_per_rank",
        "rank_local_replicated_kernel",
        "data_parallel_replicated",
    }
)
INCOMPLETE_DISTRIBUTION_SEMANTICS = frozenset(
    {
        "hybrid_sharded_forward_with_replicated_state",
        "slice_parallel_state_with_local_facade",
        "manual_sliced_tensor_contraction",
        "observable_term_parallel",
        "requires_runtime_summary",
    }
)

__all__ = (
    "CLAIMABILITY_STATUSES",
    "CLAIM_EVIDENCE_TYPES",
    "EVIDENCE_BACKEND_FAMILIES",
    "INCOMPLETE_DISTRIBUTION_SEMANTICS",
    "MULTI_NODE_TRANSPORT_BACKENDS",
    "MPS_BACKWARD_READINESS_STATUSES",
    "MPS_STATE_MODES",
    "RELEASE_CLAIM_EVIDENCE_TYPES",
    "REPLICATED_DISTRIBUTION_SEMANTICS",
    "SCALABLE_DISTRIBUTION_SEMANTICS",
    "SINGLE_DEVICE_DISTRIBUTION_SEMANTICS",
    "STATEVECTOR_STATE_MODES",
    "STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES",
    "TRANSPORT_EVIDENCE_STATUSES",
)
