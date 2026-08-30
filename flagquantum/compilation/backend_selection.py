"""Cost-aware selection between dense, MPS, and tensor-network simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

from ..core.ir import CircuitIR
from .estimates import estimate_mps_bytes, estimate_state_bytes
from .tn_calibration import TNWorkingSetCalibration

OutputTarget = Literal[
    "full_state",
    "expectation",
    "local_observables",
    "single_amplitude",
    "few_amplitudes",
    "samples",
]


@dataclass(frozen=True)
class BackendCost:
    """One auditable backend estimate."""

    backend: str
    available: bool
    estimated_memory_bytes: int
    score: int
    reasons: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "score": self.score,
            "reasons": self.reasons,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class BackendSelection:
    """Backend decision and the evidence used to make it."""

    selected_backend: str
    target: OutputTarget
    target_count: int
    require_gradients: bool
    interaction_width_proxy: int
    nearest_neighbour_fraction: float
    estimated_mps_bond: int
    tn_calibration_identity: str | None
    tn_working_set_safety_factor: float
    candidates: tuple[BackendCost, ...]
    warnings: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "selected_backend": self.selected_backend,
            "target": self.target,
            "target_count": self.target_count,
            "require_gradients": self.require_gradients,
            "interaction_width_proxy": self.interaction_width_proxy,
            "nearest_neighbour_fraction": self.nearest_neighbour_fraction,
            "estimated_mps_bond": self.estimated_mps_bond,
            "tn_calibration_identity": self.tn_calibration_identity,
            "tn_working_set_safety_factor": self.tn_working_set_safety_factor,
            "warnings": self.warnings,
            "candidates": tuple(item.summary() for item in self.candidates),
        }


def _interaction_metrics(ir: CircuitIR) -> tuple[int, float, int]:
    graph: dict[int, set[int]] = {wire: set() for wire in range(ir.n_wires)}
    local_edges = 0
    edges = 0
    crossing_counts = [0] * max(0, ir.n_wires - 1)
    for instruction in ir:
        wires = tuple(dict.fromkeys(instruction.wires))
        if len(wires) >= 2:
            left, right = min(wires), max(wires)
            for cut in range(left, right):
                crossing_counts[cut] += 1
        for index, left in enumerate(wires):
            for right in wires[index + 1 :]:
                if right not in graph[left]:
                    graph[left].add(right)
                    graph[right].add(left)
                    edges += 1
                    local_edges += int(abs(left - right) == 1)

    # Deterministic min-fill elimination. This is deliberately a cheap structural
    # proxy; a production TN path planner remains the authority for hard budgets.
    width = 0
    remaining = {node: set(neighbours) for node, neighbours in graph.items()}
    while remaining:

        def key(node: int) -> tuple[int, int, int]:
            neighbours = remaining[node]
            missing = sum(
                right not in remaining[left]
                for offset, left in enumerate(neighbours)
                for right in tuple(neighbours)[offset + 1 :]
            )
            return missing, len(neighbours), node

        node = min(remaining, key=key)
        neighbours = remaining.pop(node)
        width = max(width, len(neighbours))
        for left in neighbours:
            remaining[left].discard(node)
        for left in neighbours:
            remaining[left].update(neighbours - {left})
    return (
        width,
        (local_edges / edges if edges else 1.0),
        max(crossing_counts, default=0),
    )


def select_backend_by_cost(
    circuit_or_ir: Any,
    *,
    target: OutputTarget = "full_state",
    require_gradients: bool = False,
    world_size: int = 1,
    bsz: int = 1,
    complex_bytes: int = 8,
    memory_limit_bytes: int | None = None,
    max_bond: int | None = None,
    allow_approximate: bool = True,
    requested_backend: str = "auto",
    target_count: int = 1,
    tn_memory_calibration: TNWorkingSetCalibration | None = None,
    require_tn_memory_calibration: bool = False,
    accelerator_name: str | None = None,
) -> BackendSelection:
    """Select a representation from output needs, structure, and capacity.

    ``memory_limit_bytes`` is a per-rank/device budget. Explicit backend requests
    are preserved, but the returned warning makes predicted degradation visible.
    """

    if target not in {
        "full_state",
        "expectation",
        "local_observables",
        "single_amplitude",
        "few_amplitudes",
        "samples",
    }:
        raise ValueError(f"unsupported output target: {target!r}")
    if int(target_count) < 1:
        raise ValueError("target_count must be >= 1")
    aliases = {
        "tn": "tensor_network",
        "distributed_tensor_network": "tensor_network",
        "distributed_statevector": "statevector",
        "adaptive_mps": "mps",
        "distributed_mps": "mps",
    }
    requested_backend = aliases.get(requested_backend, requested_backend)
    if requested_backend not in {"auto", "statevector", "mps", "tensor_network"}:
        raise ValueError(f"unsupported requested backend: {requested_backend!r}")

    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    width, locality, max_cut_gates = _interaction_metrics(ir)
    ranks = max(1, int(world_size))
    dense_state = estimate_state_bytes(ir.n_wires, bsz=bsz, complex_bytes=complex_bytes)
    dense_memory = (dense_state + ranks - 1) // ranks
    # Reverse mode normally needs the primal state plus adjoint/work buffers.
    if require_gradients:
        dense_memory *= 3
    limit = int(memory_limit_bytes) if memory_limit_bytes is not None else None
    dense_fits = limit is None or dense_memory <= limit
    sparse_target = target in {
        "expectation",
        "local_observables",
        "single_amplitude",
        "few_amplitudes",
    }
    sparse_batch_limit = 1024 if target == "few_amplitudes" else 256
    if (
        target in {"few_amplitudes", "local_observables"}
        and int(target_count) > sparse_batch_limit
    ):
        sparse_target = False

    sv_blockers = () if dense_fits else ("dense_state_exceeds_per_rank_budget",)
    sv_score = 100 if dense_fits else 15
    if target == "full_state":
        sv_score += 30
    if require_gradients:
        sv_score += 15

    estimated_mps_bond = (
        int(max_bond)
        if max_bond is not None
        else 1 << min(max_cut_gates, max(0, ir.n_wires // 2))
    )
    mps_memory = estimate_mps_bytes(
        ir.n_wires,
        bsz=bsz,
        max_bond=estimated_mps_bond,
        complex_bytes=complex_bytes,
    )
    # Reverse execution retains the primal MPS plus left/right environments
    # and factorization work buffers. This is a capacity estimate, not a tape
    # guarantee; the runtime checkpoint preflight remains authoritative.
    if require_gradients:
        mps_memory *= 3
    mps_fits = limit is None or mps_memory <= limit
    mps_structured = locality >= 0.85
    mps_blockers: tuple[str, ...] = ()
    if not mps_fits:
        mps_blockers += ("mps_estimate_exceeds_per_rank_budget",)
    if target == "full_state" and not dense_fits:
        mps_blockers += ("mps_cannot_satisfy_full_state_without_dense_materialization",)
    if max_bond is None and not mps_structured:
        mps_blockers += ("unbounded_bond_growth_on_nonlocal_topology",)
    if max_bond is not None and not allow_approximate:
        mps_blockers += ("bounded_mps_requires_approximation_permission",)
    mps_score = 75 + (20 if mps_structured else -25)
    if max_bond is not None:
        # A bond cap is an explicit representation control, not merely a hint.
        mps_score += 100

    # A width proxy is not a path guarantee. It is sufficient to reject obvious
    # mismatches; execution must still run the TN path planner and memory preflight.
    tn_proxy_memory = max(
        1,
        (1 << min(width + (2 if require_gradients else 0), 62))
        * int(complex_bytes)
        * max(1, ir.n_wires)
        * (
            int(target_count)
            if target in {"few_amplitudes", "local_observables"}
            else 1
        ),
    )
    matching_tn_calibration = (
        tn_memory_calibration
        if tn_memory_calibration is not None
        and tn_memory_calibration.applies_to(
            accelerator_name=accelerator_name,
            complex_bytes=complex_bytes,
            world_size=ranks,
        )
        else None
    )
    calibration_matches = matching_tn_calibration is not None
    tn_safety_factor = (
        matching_tn_calibration.recommended_safety_factor
        if matching_tn_calibration is not None
        else 1.0
    )
    tn_memory = (
        matching_tn_calibration.apply(tn_proxy_memory)
        if matching_tn_calibration is not None
        else ceil(tn_proxy_memory * tn_safety_factor)
    )
    tn_fits = limit is None or tn_memory <= limit
    tn_blockers: tuple[str, ...] = ()
    if not sparse_target:
        tn_blockers += ("tensor_network_requires_sparse_output_target",)
    if target == "few_amplitudes" and int(target_count) > 1024:
        tn_blockers += ("amplitude_target_batch_too_large_for_sparse_tn_path",)
    if target == "local_observables" and int(target_count) > 256:
        tn_blockers += ("observable_batch_too_large_for_shared_tn_path",)
    if width > 16:
        tn_blockers += ("interaction_width_proxy_too_large",)
    if not tn_fits:
        tn_blockers += ("tn_proxy_exceeds_per_rank_budget",)
    if tn_memory_calibration is not None and not calibration_matches:
        tn_blockers += ("tn_memory_calibration_scope_mismatch",)
    if require_tn_memory_calibration and tn_memory_calibration is None:
        tn_blockers += ("tn_memory_calibration_required",)
    if dense_fits:
        tn_blockers += ("dense_statevector_is_lower_risk_for_this_capacity",)
    tn_score = 65 + (30 if sparse_target else -40) - 3 * width
    if require_gradients:
        tn_score -= 15

    candidates = (
        BackendCost(
            "statevector",
            not sv_blockers,
            dense_memory,
            sv_score,
            reasons=("exact_dense_fast_path",),
            blockers=sv_blockers,
        ),
        BackendCost(
            "mps",
            not mps_blockers,
            mps_memory,
            mps_score,
            reasons=(
                "one_dimensional_locality" if mps_structured else "bounded_bond_path",
            ),
            blockers=mps_blockers,
        ),
        BackendCost(
            "tensor_network",
            not tn_blockers,
            tn_memory,
            tn_score,
            reasons=(
                "sparse_output_contraction",
                "path_preflight_required",
                (
                    "reserved_memory_calibration_applied"
                    if calibration_matches
                    else "uncalibrated_tn_memory_proxy"
                ),
            ),
            blockers=tn_blockers,
        ),
    )
    warnings: tuple[str, ...] = ()
    if requested_backend != "auto":
        selected = requested_backend
        forced = next(item for item in candidates if item.backend == selected)
        if not forced.available:
            warnings = ("forced_backend_has_predicted_degradation_or_blockers",)
    else:
        available = tuple(item for item in candidates if item.available)
        if not available and target == "full_state":
            # No compressed representation can satisfy a full-state contract;
            # retain the exact dense/sharded path and let its hard preflight fail.
            selected = "statevector"
        else:
            selected = max(available or candidates, key=lambda item: item.score).backend

    return BackendSelection(
        selected_backend=selected,
        target=target,
        target_count=int(target_count),
        require_gradients=require_gradients,
        interaction_width_proxy=width,
        nearest_neighbour_fraction=locality,
        estimated_mps_bond=estimated_mps_bond,
        tn_calibration_identity=(
            matching_tn_calibration.identity
            if matching_tn_calibration is not None
            else None
        ),
        tn_working_set_safety_factor=tn_safety_factor,
        candidates=candidates,
        warnings=warnings,
    )


__all__ = ["BackendCost", "BackendSelection", "OutputTarget", "select_backend_by_cost"]
