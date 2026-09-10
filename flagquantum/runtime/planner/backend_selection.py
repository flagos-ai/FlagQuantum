"""Runtime cost selection between dense, MPS, and tensor-network simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

from ...core.ir import CircuitIR
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


def _min_fill_width(graph: dict[int, set[int]]) -> int:
    """Estimate contraction width with deterministic min-fill elimination."""

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
    return width


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

    return (
        _min_fill_width(graph),
        (local_edges / edges if edges else 1.0),
        max(crossing_counts, default=0),
    )


def _validated_output_target(target: str) -> OutputTarget:
    """Validate an output name and retain its finite target type."""
    supported_targets: tuple[OutputTarget, ...] = (
        "full_state",
        "expectation",
        "local_observables",
        "single_amplitude",
        "few_amplitudes",
        "samples",
    )
    for supported in supported_targets:
        if target == supported:
            return supported
    raise ValueError(f"unsupported output target: {target!r}")


def _normalized_requested_backend(requested_backend: str) -> str:
    aliases = {
        "tn": "tensor_network",
        "distributed_tensor_network": "tensor_network",
        "distributed_statevector": "statevector",
        "adaptive_mps": "mps",
        "distributed_mps": "mps",
    }
    normalized = aliases.get(requested_backend, requested_backend)
    if normalized not in {"auto", "statevector", "mps", "tensor_network"}:
        raise ValueError(f"unsupported requested backend: {normalized!r}")
    return normalized


def _select_candidate(
    candidates: tuple[BackendCost, ...],
    *,
    requested_backend: str,
    target: OutputTarget,
) -> tuple[str, tuple[str, ...]]:
    if requested_backend != "auto":
        forced = next(item for item in candidates if item.backend == requested_backend)
        warnings = (
            ()
            if forced.available
            else ("forced_backend_has_predicted_degradation_or_blockers",)
        )
        return requested_backend, warnings

    available = tuple(item for item in candidates if item.available)
    if not available and target == "full_state":
        # No compressed representation can satisfy a full-state contract;
        # retain the exact dense/sharded path and let its hard preflight fail.
        return "statevector", ()
    selected = max(available or candidates, key=lambda item: item.score)
    return selected.backend, ()


def _statevector_candidate(
    ir: CircuitIR,
    *,
    target: OutputTarget,
    require_gradients: bool,
    ranks: int,
    bsz: int,
    complex_bytes: int,
    memory_limit_bytes: int | None,
) -> tuple[BackendCost, bool]:
    dense_state = estimate_state_bytes(ir.n_wires, bsz=bsz, complex_bytes=complex_bytes)
    memory = (dense_state + ranks - 1) // ranks
    if require_gradients:
        # Reverse mode normally needs the primal state plus adjoint/work buffers.
        memory *= 3
    fits = memory_limit_bytes is None or memory <= memory_limit_bytes
    blockers = () if fits else ("dense_state_exceeds_per_rank_budget",)
    score = 100 if fits else 15
    score += 30 if target == "full_state" else 0
    score += 15 if require_gradients else 0
    return (
        BackendCost(
            "statevector",
            not blockers,
            memory,
            score,
            reasons=("exact_dense_fast_path",),
            blockers=blockers,
        ),
        fits,
    )


def _mps_candidate(
    ir: CircuitIR,
    *,
    target: OutputTarget,
    require_gradients: bool,
    locality: float,
    max_cut_gates: int,
    bsz: int,
    complex_bytes: int,
    memory_limit_bytes: int | None,
    max_bond: int | None,
    allow_approximate: bool,
    dense_fits: bool,
) -> tuple[BackendCost, int]:
    estimated_bond = (
        int(max_bond)
        if max_bond is not None
        else 1 << min(max_cut_gates, max(0, ir.n_wires // 2))
    )
    memory = estimate_mps_bytes(
        ir.n_wires,
        bsz=bsz,
        max_bond=estimated_bond,
        complex_bytes=complex_bytes,
    )
    if require_gradients:
        # Includes primal MPS, environments, and factorization work buffers.
        memory *= 3
    fits = memory_limit_bytes is None or memory <= memory_limit_bytes
    structured = locality >= 0.85
    blockers: tuple[str, ...] = ()
    if not fits:
        blockers += ("mps_estimate_exceeds_per_rank_budget",)
    if target == "full_state" and not dense_fits:
        blockers += ("mps_cannot_satisfy_full_state_without_dense_materialization",)
    if max_bond is None and not structured:
        blockers += ("unbounded_bond_growth_on_nonlocal_topology",)
    if max_bond is not None and not allow_approximate:
        blockers += ("bounded_mps_requires_approximation_permission",)
    score = 75 + (20 if structured else -25)
    if max_bond is not None:
        score += 100
    return (
        BackendCost(
            "mps",
            not blockers,
            memory,
            score,
            reasons=(
                "one_dimensional_locality" if structured else "bounded_bond_path",
            ),
            blockers=blockers,
        ),
        estimated_bond,
    )


def _tensor_network_candidate(
    ir: CircuitIR,
    *,
    target: OutputTarget,
    target_count: int,
    require_gradients: bool,
    width: int,
    ranks: int,
    complex_bytes: int,
    memory_limit_bytes: int | None,
    dense_fits: bool,
    calibration: TNWorkingSetCalibration | None,
    require_calibration: bool,
    accelerator_name: str | None,
) -> tuple[BackendCost, str | None, float]:
    sparse_target = target in {
        "expectation",
        "local_observables",
        "single_amplitude",
        "few_amplitudes",
    }
    batch_limit = 1024 if target == "few_amplitudes" else 256
    if target in {"few_amplitudes", "local_observables"} and target_count > batch_limit:
        sparse_target = False

    proxy_memory = max(
        1,
        (1 << min(width + (2 if require_gradients else 0), 62))
        * int(complex_bytes)
        * max(1, ir.n_wires)
        * (target_count if target in {"few_amplitudes", "local_observables"} else 1),
    )
    matching_calibration = (
        calibration
        if calibration is not None
        and calibration.applies_to(
            accelerator_name=accelerator_name,
            complex_bytes=complex_bytes,
            world_size=ranks,
        )
        else None
    )
    calibration_matches = matching_calibration is not None
    safety_factor = (
        matching_calibration.recommended_safety_factor
        if matching_calibration is not None
        else 1.0
    )
    memory = (
        matching_calibration.apply(proxy_memory)
        if matching_calibration is not None
        else ceil(proxy_memory * safety_factor)
    )
    fits = memory_limit_bytes is None or memory <= memory_limit_bytes
    blockers: tuple[str, ...] = ()
    if not sparse_target:
        blockers += ("tensor_network_requires_sparse_output_target",)
    if target == "few_amplitudes" and target_count > 1024:
        blockers += ("amplitude_target_batch_too_large_for_sparse_tn_path",)
    if target == "local_observables" and target_count > 256:
        blockers += ("observable_batch_too_large_for_shared_tn_path",)
    if width > 16:
        blockers += ("interaction_width_proxy_too_large",)
    if not fits:
        blockers += ("tn_proxy_exceeds_per_rank_budget",)
    if calibration is not None and not calibration_matches:
        blockers += ("tn_memory_calibration_scope_mismatch",)
    if require_calibration and calibration is None:
        blockers += ("tn_memory_calibration_required",)
    if dense_fits:
        blockers += ("dense_statevector_is_lower_risk_for_this_capacity",)
    score = 65 + (30 if sparse_target else -40) - 3 * width
    score -= 15 if require_gradients else 0
    candidate = BackendCost(
        "tensor_network",
        not blockers,
        memory,
        score,
        reasons=(
            "sparse_output_contraction",
            "path_preflight_required",
            (
                "reserved_memory_calibration_applied"
                if calibration_matches
                else "uncalibrated_tn_memory_proxy"
            ),
        ),
        blockers=blockers,
    )
    identity = (
        matching_calibration.identity if matching_calibration is not None else None
    )
    return candidate, identity, safety_factor


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

    _validated_output_target(target)
    if int(target_count) < 1:
        raise ValueError("target_count must be >= 1")
    requested_backend = _normalized_requested_backend(requested_backend)

    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    width, locality, max_cut_gates = _interaction_metrics(ir)
    ranks = max(1, int(world_size))
    limit = int(memory_limit_bytes) if memory_limit_bytes is not None else None
    statevector, dense_fits = _statevector_candidate(
        ir,
        target=target,
        require_gradients=require_gradients,
        ranks=ranks,
        bsz=bsz,
        complex_bytes=complex_bytes,
        memory_limit_bytes=limit,
    )
    mps, estimated_mps_bond = _mps_candidate(
        ir,
        target=target,
        require_gradients=require_gradients,
        locality=locality,
        max_cut_gates=max_cut_gates,
        bsz=bsz,
        complex_bytes=complex_bytes,
        memory_limit_bytes=limit,
        max_bond=max_bond,
        allow_approximate=allow_approximate,
        dense_fits=dense_fits,
    )
    tensor_network, calibration_identity, tn_safety_factor = _tensor_network_candidate(
        ir,
        target=target,
        target_count=int(target_count),
        require_gradients=require_gradients,
        width=width,
        ranks=ranks,
        complex_bytes=complex_bytes,
        memory_limit_bytes=limit,
        dense_fits=dense_fits,
        calibration=tn_memory_calibration,
        require_calibration=require_tn_memory_calibration,
        accelerator_name=accelerator_name,
    )
    candidates = (statevector, mps, tensor_network)
    selected, warnings = _select_candidate(
        candidates,
        requested_backend=requested_backend,
        target=target,
    )

    return BackendSelection(
        selected_backend=selected,
        target=target,
        target_count=int(target_count),
        require_gradients=require_gradients,
        interaction_width_proxy=width,
        nearest_neighbour_fraction=locality,
        estimated_mps_bond=estimated_mps_bond,
        tn_calibration_identity=calibration_identity,
        tn_working_set_safety_factor=tn_safety_factor,
        candidates=candidates,
        warnings=warnings,
    )


__all__ = ["BackendCost", "BackendSelection", "OutputTarget", "select_backend_by_cost"]
