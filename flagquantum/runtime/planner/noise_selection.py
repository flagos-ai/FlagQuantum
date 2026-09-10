"""Auditable Runtime backend selection for noisy simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log, sqrt
from typing import Any, Mapping

from ...compiler import lower_noise_model
from ...core.ir import ensure_circuit_ir
from .estimates import estimate_density_bytes, estimate_mps_bytes, estimate_state_bytes
from .noise_calibration import (
    NoiseSelectorCalibration,
    load_noise_selector_calibration,
)


@dataclass(frozen=True)
class _TrajectoryEstimate:
    variance: float
    required: int | None
    feasible_within_cap: bool | None
    cost_count: int | None
    basis: str | None

    def metadata(
        self,
        *,
        trajectory_cap: int | None,
        pilot_variance: float | None,
        pilot_trajectories: int | None,
        pilot_confidence_level: float | None,
        pilot_observable_count: int,
    ) -> dict[str, Any]:
        return {
            "trajectory_cap": trajectory_cap,
            "estimated_trajectories_to_target": self.required,
            "trajectory_estimate_basis": self.basis,
            "trajectory_variance_estimate": (
                self.variance if self.required is not None else None
            ),
            "pilot_trajectories": pilot_trajectories,
            "pilot_sample_variance": pilot_variance,
            "pilot_confidence_level": pilot_confidence_level,
            "pilot_observable_count": (
                pilot_observable_count if pilot_variance is not None else None
            ),
            "target_feasible_within_cap": self.feasible_within_cap,
        }


@dataclass(frozen=True)
class NoiseBackendCandidate:
    """One noisy backend option with explicit acceptance or rejection evidence."""

    mode: str
    representation: str
    evolution: str
    estimated_memory_bytes: int
    fits_memory: bool
    eligible: bool
    exact_quantum_channel: bool
    sampling_error: bool
    truncation_error: bool
    reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "representation": self.representation,
            "evolution": self.evolution,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "fits_memory": self.fits_memory,
            "eligible": self.eligible,
            "exact_quantum_channel": self.exact_quantum_channel,
            "sampling_error": self.sampling_error,
            "truncation_error": self.truncation_error,
            "reasons": self.reasons,
            "rejection_reasons": self.rejection_reasons,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class NoiseExecutionSelection:
    """Selected noisy backend plus all evaluated alternatives."""

    selected_mode: str
    selected_candidate: NoiseBackendCandidate
    candidates: tuple[NoiseBackendCandidate, ...]
    noise_event_count: int
    maximum_kraus_rank: int
    memory_limit_bytes: int | None
    selection_basis: str = "analytic_policy"
    calibration_device: str | None = None
    target_standard_error: float | None = None
    estimated_trajectories_to_target: int | None = None
    target_feasible_within_cap: bool | None = None
    trajectory_variance_estimate: float | None = None
    pilot_trajectories: int | None = None
    pilot_sample_variance: float | None = None
    pilot_confidence_level: float | None = None
    pilot_observable_count: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "planner": "noise_execution_selection_v1",
            "selected_mode": self.selected_mode,
            "noise_event_count": self.noise_event_count,
            "maximum_kraus_rank": self.maximum_kraus_rank,
            "memory_limit_bytes": self.memory_limit_bytes,
            "selection_basis": self.selection_basis,
            "calibration_device": self.calibration_device,
            "target_standard_error": self.target_standard_error,
            "estimated_trajectories_to_target": self.estimated_trajectories_to_target,
            "target_feasible_within_cap": self.target_feasible_within_cap,
            "trajectory_variance_estimate": self.trajectory_variance_estimate,
            "pilot_trajectories": self.pilot_trajectories,
            "pilot_sample_variance": self.pilot_sample_variance,
            "pilot_confidence_level": self.pilot_confidence_level,
            "pilot_observable_count": self.pilot_observable_count,
            "selected_candidate": self.selected_candidate.summary(),
            "candidates": tuple(item.summary() for item in self.candidates),
        }


def _validate_selection_inputs(
    *,
    trajectories: int | None,
    min_trajectories: int,
    target_standard_error: float | None,
    pilot_variance: float | None,
    pilot_trajectories: int | None,
    pilot_confidence_level: float | None,
    pilot_observable_count: int,
) -> None:
    if min_trajectories <= 0:
        raise ValueError("min_trajectories must be positive")
    if target_standard_error is not None and target_standard_error <= 0:
        raise ValueError("target_standard_error must be positive")
    if target_standard_error is not None and trajectories is None:
        raise ValueError(
            "target_standard_error requires an explicit trajectory ceiling"
        )
    if pilot_variance is not None and (
        not isfinite(pilot_variance) or pilot_variance < 0.0
    ):
        raise ValueError("pilot_variance must be finite and non-negative")
    if pilot_variance is not None and (
        pilot_trajectories is None or pilot_trajectories < 2
    ):
        raise ValueError("pilot_variance requires at least two pilot trajectories")
    if pilot_variance is None and pilot_trajectories is not None:
        raise ValueError("pilot_trajectories requires pilot_variance")
    if pilot_confidence_level is not None and pilot_variance is None:
        raise ValueError("pilot_confidence_level requires pilot_variance")
    if pilot_confidence_level is not None and not 0.0 < pilot_confidence_level < 1.0:
        raise ValueError("pilot_confidence_level must be between zero and one")
    if pilot_observable_count <= 0:
        raise ValueError("pilot_observable_count must be positive")


def _estimate_trajectory_requirement(
    *,
    trajectories: int | None,
    min_trajectories: int,
    target_standard_error: float | None,
    pilot_variance: float | None,
    pilot_trajectories: int | None,
    pilot_confidence_level: float | None,
    pilot_observable_count: int,
) -> _TrajectoryEstimate:
    variance = 1.0 if pilot_variance is None else pilot_variance
    if pilot_confidence_level is not None:
        assert pilot_variance is not None
        assert pilot_trajectories is not None
        delta = (1.0 - pilot_confidence_level) / pilot_observable_count
        standard_deviation_radius = 2.0 * sqrt(
            2.0 * log(1.0 / delta) / (pilot_trajectories - 1)
        )
        variance = min(
            1.0,
            (sqrt(max(pilot_variance, 0.0)) + standard_deviation_radius) ** 2,
        )
    required = (
        None
        if target_standard_error is None
        else max(min_trajectories, ceil(variance / target_standard_error**2))
    )
    feasible = (
        None if required is None or trajectories is None else trajectories >= required
    )
    cost_count = (
        trajectories
        if required is None or trajectories is None
        else min(trajectories, required)
    )
    basis = None
    if required is not None:
        if pilot_confidence_level is not None:
            basis = "pilot_variance_upper_confidence_bound"
        elif pilot_variance is not None:
            basis = "pilot_variance"
        else:
            basis = "bounded_pauli_variance_le_one"
    return _TrajectoryEstimate(variance, required, feasible, cost_count, basis)


def _calibration_metadata(
    calibration: NoiseSelectorCalibration | None,
    *,
    mode: str,
    n_wires: int,
    channel_count: int,
    circuit_digest: str,
    noise_model_identity: str,
    trajectory_count: int | None,
    trajectory_batch_size: int,
    world_size: int,
) -> dict[str, Any]:
    if calibration is None:
        return {}
    match = calibration.estimate_seconds(
        mode=mode,
        n_wires=n_wires,
        channel_count=channel_count,
        circuit_digest=circuit_digest,
        noise_model_identity=noise_model_identity,
        trajectories=trajectory_count,
        trajectory_batch_size=trajectory_batch_size,
        world_size=world_size,
    )
    if match is None:
        return {"calibration_match": False}
    seconds, record = match
    return {
        "calibration_match": True,
        "calibrated_estimated_seconds": seconds,
        "calibration_record": {
            "device_name": calibration.device_name,
            "torch_version": calibration.torch_version,
            "world_size": calibration.world_size,
            "depth": record.depth,
            "median_seconds": record.median_seconds,
            "executed_trajectories": record.executed_trajectories,
        },
    }


def _density_candidate(
    *,
    estimated_memory_bytes: int,
    fits_memory: bool,
    trajectories: int | None,
    target_standard_error: float | None,
    calibration_metadata: Mapping[str, Any],
) -> NoiseBackendCandidate:
    rejections = []
    if not fits_memory:
        rejections.append("memory_limit_exceeded")
    if trajectories is not None and target_standard_error is None:
        rejections.append("trajectory_execution_explicitly_requested")
    return NoiseBackendCandidate(
        mode="density_matrix",
        representation="density_matrix",
        evolution="exact_channel",
        estimated_memory_bytes=estimated_memory_bytes,
        fits_memory=fits_memory,
        eligible=not rejections,
        exact_quantum_channel=True,
        sampling_error=False,
        truncation_error=False,
        reasons=("small_system_correctness_oracle",),
        rejection_reasons=tuple(rejections),
        metadata=calibration_metadata,
    )


def _statevector_candidate(
    *,
    estimated_memory_bytes: int,
    fits_memory: bool,
    trajectories: int | None,
    exact_noise: bool,
    trajectory_batch_size: int,
    workspace_factor: int,
    world_size: int,
    trajectory_metadata: Mapping[str, Any],
    calibration_metadata: Mapping[str, Any],
) -> NoiseBackendCandidate:
    rejections = []
    if trajectories is None:
        rejections.append("trajectory_count_not_requested")
    if exact_noise:
        rejections.append("exact_noise_required")
    if not fits_memory:
        rejections.append("memory_limit_exceeded")
    return NoiseBackendCandidate(
        mode="noisy_statevector",
        representation="statevector",
        evolution="quantum_trajectory",
        estimated_memory_bytes=estimated_memory_bytes,
        fits_memory=fits_memory,
        eligible=not rejections,
        exact_quantum_channel=False,
        sampling_error=True,
        truncation_error=False,
        reasons=("general_dense_circuit_trajectory_path",),
        rejection_reasons=tuple(rejections),
        metadata={
            "trajectory_batch_size": trajectory_batch_size,
            "workspace_factor": workspace_factor,
            "world_size": world_size,
            **trajectory_metadata,
            **calibration_metadata,
        },
    )


def _mps_candidate(
    *,
    estimated_memory_bytes: int,
    fits_memory: bool,
    exact_noise: bool,
    allow_approximate: bool,
    world_size: int,
    max_bond: int | None,
    cutoff: float,
    trajectory_metadata: Mapping[str, Any],
    calibration_metadata: Mapping[str, Any],
) -> NoiseBackendCandidate:
    rejections = []
    if exact_noise:
        rejections.append("exact_noise_required")
    if not allow_approximate:
        rejections.append("approximate_execution_disabled")
    if not fits_memory:
        rejections.append("memory_limit_exceeded")
    if world_size > 1:
        rejections.append("distributed_collective_not_implemented")
    return NoiseBackendCandidate(
        mode="noisy_mps",
        representation="mps",
        evolution="quantum_trajectory",
        estimated_memory_bytes=estimated_memory_bytes,
        fits_memory=fits_memory,
        eligible=not rejections,
        exact_quantum_channel=False,
        sampling_error=True,
        truncation_error=True,
        reasons=(
            (
                "explicit_mps_controls"
                if max_bond is not None or cutoff > 0
                else "compressed_state_fallback"
            ),
        ),
        rejection_reasons=tuple(rejections),
        metadata={
            "max_bond": max_bond,
            "cutoff": cutoff,
            "world_size": world_size,
            "distributed_public_execution_supported": world_size == 1,
            **trajectory_metadata,
            **calibration_metadata,
        },
    )


def _select_candidate(
    candidates: tuple[NoiseBackendCandidate, ...],
    *,
    exact_noise: bool,
    max_bond: int | None,
    cutoff: float,
    trajectories: int | None,
    target_standard_error: float | None,
) -> tuple[NoiseBackendCandidate, str]:
    density, statevector, mps = candidates
    selection_basis = "analytic_policy"
    if exact_noise:
        selected = density
    elif max_bond is not None or cutoff > 0:
        selected = mps
    elif trajectories is not None and statevector.eligible:
        pool = candidates if target_standard_error is not None else (statevector, mps)
        eligible = tuple(candidate for candidate in pool if candidate.eligible)
        calibrated = tuple(
            candidate
            for candidate in eligible
            if "calibrated_estimated_seconds" in (candidate.metadata or {})
        )
        if calibrated and len(calibrated) == len(eligible):
            selected = min(
                calibrated,
                key=lambda candidate: float(
                    (candidate.metadata or {})["calibrated_estimated_seconds"]
                ),
            )
            selection_basis = "exact_device_calibration"
        else:
            selected = statevector
    elif trajectories is None and density.eligible:
        selected = density
    else:
        selected = mps

    if selected.eligible:
        return selected, selection_basis
    if exact_noise or max_bond is not None or cutoff > 0:
        raise ValueError(
            f"requested noisy execution candidate {selected.mode!r} is not "
            f"eligible: {selected.rejection_reasons}"
        )
    available = tuple(candidate for candidate in candidates if candidate.eligible)
    if not available:
        evidence = {
            candidate.mode: candidate.rejection_reasons for candidate in candidates
        }
        raise ValueError(f"no noisy execution candidate is eligible: {evidence}")
    return available[0], selection_basis


def plan_noise_execution_selection(
    circuit_or_ir: Any,
    noise_model: Any,
    *,
    bsz: int = 1,
    world_size: int = 1,
    complex_bytes: int = 8,
    memory_limit_bytes: int | None = None,
    trajectories: int | None = None,
    trajectory_batch_size: int = 32,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    allow_approximate: bool = True,
    exact_noise: bool = False,
    calibration: Any | None = None,
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    pilot_variance: float | None = None,
    pilot_trajectories: int | None = None,
    pilot_confidence_level: float | None = None,
    pilot_observable_count: int = 1,
) -> NoiseExecutionSelection:
    """Evaluate density, statevector-trajectory, and MPS-trajectory candidates."""

    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_selection_inputs(
        trajectories=trajectories,
        min_trajectories=min_trajectories,
        target_standard_error=target_standard_error,
        pilot_variance=pilot_variance,
        pilot_trajectories=pilot_trajectories,
        pilot_confidence_level=pilot_confidence_level,
        pilot_observable_count=pilot_observable_count,
    )
    trajectory_estimate = _estimate_trajectory_requirement(
        trajectories=trajectories,
        min_trajectories=min_trajectories,
        target_standard_error=target_standard_error,
        pilot_variance=pilot_variance,
        pilot_trajectories=pilot_trajectories,
        pilot_confidence_level=pilot_confidence_level,
        pilot_observable_count=pilot_observable_count,
    )
    lowered = lower_noise_model(ir, noise_model)
    channels = tuple(
        instruction
        for instruction in lowered.instructions
        if instruction.metadata.get("is_channel")
    )

    def kraus_rank(channel: Any) -> int:
        if channel.matrix is None:
            raise ValueError("lowered noise channel is missing Kraus operators")
        return len(channel.matrix)

    maximum_kraus_rank = max((kraus_rank(item) for item in channels), default=1)
    selected_calibration: NoiseSelectorCalibration | None = (
        load_noise_selector_calibration(calibration)
        if calibration is not None
        else None
    )

    def calibration_metadata(mode: str) -> dict[str, Any]:
        return _calibration_metadata(
            selected_calibration,
            mode=mode,
            n_wires=ir.n_wires,
            channel_count=len(channels),
            circuit_digest=ir.content_hash,
            noise_model_identity=noise_model.identity,
            trajectory_count=trajectory_estimate.cost_count,
            trajectory_batch_size=trajectory_batch_size,
            world_size=world_size,
        )

    density_bytes = estimate_density_bytes(
        ir.n_wires, bsz=bsz, complex_bytes=complex_bytes
    )
    block_count = min(int(trajectories or 32), int(trajectory_batch_size))
    state_bytes = estimate_state_bytes(
        ir.n_wires, bsz=bsz * block_count, complex_bytes=complex_bytes
    )
    workspace_factor = 2 + maximum_kraus_rank
    statevector_bytes = state_bytes * workspace_factor
    mps_bytes = estimate_mps_bytes(
        ir.n_wires,
        bsz=bsz,
        max_bond=max_bond,
        complex_bytes=complex_bytes,
    )

    trajectory_metadata = trajectory_estimate.metadata(
        trajectory_cap=trajectories,
        pilot_variance=pilot_variance,
        pilot_trajectories=pilot_trajectories,
        pilot_confidence_level=pilot_confidence_level,
        pilot_observable_count=pilot_observable_count,
    )
    fits_density = memory_limit_bytes is None or density_bytes <= memory_limit_bytes
    fits_statevector = (
        memory_limit_bytes is None or statevector_bytes <= memory_limit_bytes
    )
    fits_mps = memory_limit_bytes is None or mps_bytes <= memory_limit_bytes
    density = _density_candidate(
        estimated_memory_bytes=density_bytes,
        fits_memory=fits_density,
        trajectories=trajectories,
        target_standard_error=target_standard_error,
        calibration_metadata=calibration_metadata("density_matrix"),
    )
    statevector = _statevector_candidate(
        estimated_memory_bytes=statevector_bytes,
        fits_memory=fits_statevector,
        trajectories=trajectories,
        exact_noise=exact_noise,
        trajectory_batch_size=block_count,
        workspace_factor=workspace_factor,
        world_size=world_size,
        trajectory_metadata=trajectory_metadata,
        calibration_metadata=calibration_metadata("noisy_statevector"),
    )
    mps = _mps_candidate(
        estimated_memory_bytes=mps_bytes,
        fits_memory=fits_mps,
        exact_noise=exact_noise,
        allow_approximate=allow_approximate,
        world_size=world_size,
        max_bond=max_bond,
        cutoff=cutoff,
        trajectory_metadata=trajectory_metadata,
        calibration_metadata=calibration_metadata("noisy_mps"),
    )
    candidates = (density, statevector, mps)
    selected, selection_basis = _select_candidate(
        candidates,
        exact_noise=exact_noise,
        max_bond=max_bond,
        cutoff=cutoff,
        trajectories=trajectories,
        target_standard_error=target_standard_error,
    )
    return NoiseExecutionSelection(
        selected_mode=selected.mode,
        selected_candidate=selected,
        candidates=candidates,
        noise_event_count=len(channels),
        maximum_kraus_rank=maximum_kraus_rank,
        memory_limit_bytes=memory_limit_bytes,
        selection_basis=selection_basis,
        calibration_device=(
            selected_calibration.device_name
            if selected_calibration is not None
            else None
        ),
        target_standard_error=target_standard_error,
        estimated_trajectories_to_target=trajectory_estimate.required,
        target_feasible_within_cap=trajectory_estimate.feasible_within_cap,
        trajectory_variance_estimate=(
            trajectory_estimate.variance
            if trajectory_estimate.required is not None
            else None
        ),
        pilot_trajectories=pilot_trajectories,
        pilot_sample_variance=pilot_variance,
        pilot_confidence_level=pilot_confidence_level,
        pilot_observable_count=(
            pilot_observable_count if pilot_variance is not None else None
        ),
    )


__all__ = (
    "NoiseBackendCandidate",
    "NoiseExecutionSelection",
    "plan_noise_execution_selection",
)
