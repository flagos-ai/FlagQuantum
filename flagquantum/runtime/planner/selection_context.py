"""Normalized inputs and resource estimates for Runtime selection."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_plans import CircuitAnalysisView
from .estimates import (
    estimate_density_bytes,
    estimate_mps_bytes,
    estimate_state_bytes,
    estimate_tensor_network_bytes,
)


@dataclass(frozen=True)
class RuntimeSelectionContext:
    """Validated topology, policy labels, and memory estimates."""

    world_size: int
    local_world_size: int
    node_count: int
    prefer_distributed: bool
    requested_state_mode: str
    objective: str
    dense_bytes: int
    density_bytes: int
    mps_bytes: int
    tensor_network_peak_bytes: int
    sharded_dense_bytes: int


def build_runtime_selection_context(
    analysis: CircuitAnalysisView,
    *,
    bsz: int,
    world_size: int,
    local_world_size: int | None,
    node_count: int | None,
    complex_bytes: int,
    max_bond: int | None,
    max_intermediate_size: int | None,
    state_mode: str,
    prefer_jax: bool,
    prefer_distributed: bool | None,
    require_gradients: bool,
    require_deployment: bool,
) -> RuntimeSelectionContext:
    """Normalize selection inputs without initializing a runtime backend."""

    normalized_world_size = max(1, int(world_size))
    normalized_local_world_size = max(
        1,
        min(
            normalized_world_size,
            int(local_world_size or normalized_world_size),
        ),
    )
    normalized_node_count = max(
        1,
        (
            int(node_count)
            if node_count is not None
            else (normalized_world_size + normalized_local_world_size - 1)
            // normalized_local_world_size
        ),
    )
    distributed_preferred = (
        normalized_world_size > 1
        if prefer_distributed is None
        else bool(prefer_distributed)
    )
    objective_parts = [
        "training" if require_gradients else "inference",
        "deployment_ready" if require_deployment else "simulation",
        "jax_preferred" if prefer_jax else "native_preferred",
    ]
    if distributed_preferred:
        objective_parts.append("distributed_scale_out")

    dense_bytes = estimate_state_bytes(
        analysis.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    density_bytes = estimate_density_bytes(
        analysis.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    mps_bytes = estimate_mps_bytes(
        analysis.n_wires,
        bsz=bsz,
        max_bond=max_bond,
        complex_bytes=complex_bytes,
    )
    tensor_network_bytes = estimate_tensor_network_bytes(
        analysis.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    return RuntimeSelectionContext(
        world_size=normalized_world_size,
        local_world_size=normalized_local_world_size,
        node_count=normalized_node_count,
        prefer_distributed=distributed_preferred,
        requested_state_mode=(
            "tensor_network" if state_mode == "tn" else str(state_mode)
        ),
        objective="+".join(objective_parts),
        dense_bytes=dense_bytes,
        density_bytes=density_bytes,
        mps_bytes=mps_bytes,
        tensor_network_peak_bytes=int(max_intermediate_size or tensor_network_bytes),
        sharded_dense_bytes=max(
            1,
            (dense_bytes + normalized_world_size - 1) // normalized_world_size,
        ),
    )
