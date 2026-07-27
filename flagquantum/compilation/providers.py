"""Runtime candidate providers grouped by state representation."""

from __future__ import annotations

from typing import Any, Mapping

from .candidate_plans import CircuitAnalysisView
from .candidates import RuntimeCandidateBuilder


def add_local_state_candidates(
    builder: RuntimeCandidateBuilder,
    *,
    analysis: CircuitAnalysisView,
    world_size: int,
    dense_bytes: int,
    density_bytes: int,
    require_gradients: bool,
    noise_model_present: bool,
) -> tuple[str, ...]:
    noisy_warning = (
        ("noise_model_requires_noisy_runtime",)
        if noise_model_present or analysis.has_noise
        else ()
    )
    if not noise_model_present and not analysis.has_noise:
        builder.add(
            mode="statevector",
            state="statevector",
            backend="pytorch",
            semantics="single_device_fast_path",
            memory=dense_bytes,
            gradient="native_autograd" if require_gradients else "not_required",
            deployment=True,
            base_score=82,
            reasons=("exact_dense_reference", "best_general_local_correctness_path"),
        )
        builder.add(
            mode="jax_kernel_statevector",
            state="statevector",
            backend="jax",
            semantics=(
                "single_device_fast_path"
                if world_size == 1
                else "rank_local_replicated_kernel"
            ),
            memory=dense_bytes,
            gradient="jax_value_and_grad" if require_gradients else "not_required",
            deployment=True,
            base_score=88,
            reasons=(
                "torch_interface_with_jax_quantum_kernel",
                "single_rank_acceleration",
            ),
            warnings=(
                ("rank_local_jax_kernel_is_not_capacity_scaling",)
                if world_size > 1
                else ()
            ),
            claim_allowed=False,
        )
    else:
        builder.add(
            mode="density_matrix",
            state="density_matrix",
            backend="pytorch",
            semantics="single_device_fast_path",
            memory=density_bytes,
            gradient="native_autograd" if require_gradients else "not_required",
            deployment=True,
            base_score=86,
            reasons=("noise_model_supported", "exact_density_matrix_path"),
        )
    return noisy_warning


def add_mps_candidates(
    builder: RuntimeCandidateBuilder,
    *,
    analysis: CircuitAnalysisView,
    world_size: int,
    mps_bytes: int,
    require_gradients: bool,
    max_bond: int | None,
    cutoff: float,
    allow_approximate: bool,
    memory_limit_bytes: int | None,
    prefer_jax: bool,
    noisy_warning: tuple[str, ...],
) -> tuple[str, ...]:
    warnings: tuple[str, ...] = ()
    if analysis.multi_qubit_gates:
        warnings += ("mps_multi_qubit_gate_path_may_require_dense_local_rebuild",)
    blockers: tuple[str, ...] = ()
    if (max_bond is not None or cutoff > 0) and not allow_approximate:
        blockers = ("approximate_mps_controls_not_allowed",)
    if max_bond is not None or cutoff > 0:
        warnings += ("mps_truncation_controls_active",)
    builder.add(
        mode="mps",
        state="mps",
        backend="pytorch",
        semantics="single_device_fast_path",
        memory=mps_bytes,
        gradient="native_autograd" if require_gradients else "not_required",
        deployment=True,
        base_score=(
            80
            if max_bond is not None or cutoff > 0 or memory_limit_bytes is not None
            else 70
        ),
        reasons=("low_bond_memory_efficient_path",),
        warnings=warnings + noisy_warning,
        blockers=blockers,
    )
    builder.add(
        mode="jax_kernel_mps",
        state="mps",
        backend="jax",
        semantics=(
            "single_device_fast_path"
            if world_size == 1
            else "rank_local_replicated_kernel"
        ),
        memory=mps_bytes,
        gradient="jax_value_and_grad" if require_gradients else "not_required",
        deployment=True,
        base_score=96 if max_bond is not None or cutoff > 0 or prefer_jax else 76,
        reasons=("torch_interface_with_jax_mps_kernel", "low_bond_training_fast_path"),
        warnings=warnings
        + noisy_warning
        + (
            ("rank_local_jax_kernel_is_not_capacity_scaling",) if world_size > 1 else ()
        ),
        blockers=blockers,
        claim_allowed=False,
    )
    return warnings


def add_tensor_network_candidates(
    builder: RuntimeCandidateBuilder,
    *,
    world_size: int,
    peak_bytes: int,
    require_gradients: bool,
    prefer_jax: bool,
    noisy_warning: tuple[str, ...],
) -> None:
    peak_warning = ("tensor_network_peak_memory_depends_on_contraction_path",)
    builder.add(
        mode="tensor_network",
        state="tensor_network",
        backend="pytorch",
        semantics="single_device_fast_path",
        memory=peak_bytes,
        gradient="native_autograd" if require_gradients else "not_required",
        deployment=True,
        base_score=72,
        reasons=("general_tensor_network_path",),
        warnings=peak_warning + noisy_warning,
    )
    builder.add(
        mode="jax_kernel_tensor_network",
        state="tensor_network",
        backend="jax",
        semantics=(
            "single_device_fast_path"
            if world_size == 1
            else "rank_local_replicated_kernel"
        ),
        memory=peak_bytes,
        gradient="jax_value_and_grad" if require_gradients else "not_required",
        deployment=True,
        base_score=84 if prefer_jax else 74,
        reasons=("torch_interface_with_jax_tensor_network_kernel",),
        warnings=peak_warning
        + noisy_warning
        + (
            ("rank_local_jax_kernel_is_not_capacity_scaling",) if world_size > 1 else ()
        ),
        claim_allowed=False,
    )


def add_distributed_candidates(
    builder: RuntimeCandidateBuilder,
    *,
    world_size: int,
    sharded_dense_bytes: int,
    mps_bytes: int,
    tensor_network_peak_bytes: int,
    require_gradients: bool,
    prefer_jax: bool,
    mps_warnings: tuple[str, ...],
    statevector_summary: Mapping[str, Any] | None,
    statevector_blockers: tuple[str, ...],
    statevector_claim_allowed: bool,
    mps_summary: Mapping[str, Any] | None,
    mps_blockers: tuple[str, ...],
) -> None:
    builder.add(
        mode="distributed_statevector",
        state="statevector",
        backend="pytorch",
        semantics=(
            "sharded_across_ranks" if world_size > 1 else "single_device_fast_path"
        ),
        memory=sharded_dense_bytes,
        gradient=(
            "forward_sharded_backward_pending" if require_gradients else "forward_only"
        ),
        deployment=True,
        base_score=78,
        reasons=("amplitude_sharded_capacity_path",),
        blockers=(("world_size_must_be_greater_than_one",) if world_size <= 1 else ()),
        claim_allowed=not require_gradients,
    )
    builder.add(
        mode="jax_sharded_statevector",
        state="statevector",
        backend="jax",
        semantics=(
            "sharded_across_ranks" if world_size > 1 else "single_device_fast_path"
        ),
        memory=sharded_dense_bytes,
        gradient="jax_sharded_parameter_vjp" if require_gradients else "forward_only",
        deployment=True,
        base_score=90 if prefer_jax else 82,
        reasons=("jax_amplitude_sharded_parameter_gradient_path",),
        warnings=("production_pmap_or_shard_map_runtime_summary_required_for_claim",),
        blockers=(
            ("world_size_must_be_greater_than_one",)
            if world_size <= 1
            else statevector_blockers
        ),
        claim_allowed=statevector_claim_allowed,
        gradient_details=statevector_summary,
        jax_statevector_training_summary=statevector_summary,
    )
    builder.add(
        mode="distributed_mps",
        state="mps",
        backend="pytorch",
        semantics="requires_runtime_summary",
        intended_semantics="sharded_across_ranks",
        memory=max(1, (mps_bytes + world_size - 1) // world_size),
        gradient=(
            "distributed_backward_pending" if require_gradients else "forward_only"
        ),
        deployment=True,
        base_score=70,
        reasons=("site_or_bond_sharding_target",),
        warnings=mps_warnings,
        blockers=(
            "distributed_mps_runtime_summary_required_for_scalability_claim",
            "mps_boundary_adjoint_exchange_pending",
            "mps_parameter_gradient_ownership_pending",
            "mps_optimizer_update_ownership_pending",
        ),
        claim_allowed=False,
    )
    builder.add(
        mode="jax_sharded_mps",
        state="mps",
        backend="jax",
        semantics=str(
            mps_summary.get("distribution_semantics", "requires_runtime_summary")
            if mps_summary
            else "requires_runtime_summary"
        ),
        intended_semantics=str(
            mps_summary.get("intended_distribution_semantics", "sharded_across_ranks")
            if mps_summary
            else "sharded_across_ranks"
        ),
        memory=max(
            1,
            *(
                tuple(
                    int(item)
                    for item in mps_summary.get("local_memory_bytes_by_rank", ())
                )
                if mps_summary and mps_summary.get("local_memory_bytes_by_rank")
                else ((mps_bytes + world_size - 1) // world_size,)
            ),
        ),
        gradient=(
            "distributed_backward_pending" if require_gradients else "forward_only"
        ),
        deployment=True,
        base_score=76 if prefer_jax else 68,
        reasons=("jax_site_sharded_mps_target",),
        warnings=mps_warnings,
        blockers=(
            mps_blockers
            if require_gradients and mps_blockers
            else (
                "jax_pmap_mps_site_sharded_executor_pending",
                "mps_boundary_adjoint_exchange_pending",
                "mps_parameter_gradient_ownership_pending",
                "mps_optimizer_update_ownership_pending",
            )
        ),
        claim_allowed=False,
        gradient_details=mps_summary,
        metadata={
            "sharding_plan_available": (
                bool(mps_summary.get("sharding_plan_available", False))
                if mps_summary
                else False
            ),
        },
        jax_mps_training_summary=mps_summary,
    )
    distributed_tn_bytes = max(
        1, (tensor_network_peak_bytes + world_size - 1) // world_size
    )
    builder.add(
        mode="distributed_tensor_network",
        state="tensor_network",
        backend="pytorch",
        semantics="requires_runtime_summary",
        intended_semantics="manual_sliced_tensor_contraction",
        memory=distributed_tn_bytes,
        gradient=(
            "distributed_backward_pending" if require_gradients else "forward_only"
        ),
        deployment=True,
        base_score=68,
        reasons=("sliced_tensor_contraction_target",),
        warnings=("tensor_network_peak_memory_depends_on_contraction_path",),
        blockers=(
            "distributed_tensor_network_runtime_summary_required_for_scalability_claim",
            "distributed_tensor_network_reverse_contraction_pending",
        ),
        claim_allowed=False,
    )
    builder.add(
        mode="jax_sharded_tensor_network",
        state="tensor_network",
        backend="jax",
        semantics="manual_sliced_tensor_contraction",
        intended_semantics="manual_sliced_tensor_contraction",
        memory=distributed_tn_bytes,
        gradient="tensor_node_gradients_only" if require_gradients else "forward_only",
        deployment=True,
        base_score=80 if prefer_jax else 70,
        reasons=("jax_sliced_tensor_network_reduction_target",),
        warnings=("tensor_network_peak_memory_depends_on_contraction_path",),
        blockers=("jax_sliced_tensor_network_parameter_gate_pullback_pending",),
        claim_allowed=False,
    )


__all__ = [
    "add_distributed_candidates",
    "add_local_state_candidates",
    "add_mps_candidates",
    "add_tensor_network_candidates",
]
