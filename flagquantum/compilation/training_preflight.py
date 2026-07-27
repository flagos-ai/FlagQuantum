"""Optional JAX training preflight used by runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DistributedTrainingPreflight:
    """Evidence collected before distributed candidates are constructed."""

    statevector_summary: Mapping[str, Any] | None = None
    statevector_blockers: tuple[str, ...] = ()
    statevector_claim_allowed: bool = False
    mps_summary: Mapping[str, Any] | None = None
    mps_blockers: tuple[str, ...] = ()


def _unavailable_summary(exc: Exception) -> dict[str, str]:
    return {
        "status": "unavailable",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def collect_distributed_training_preflight(
    ir: Any,
    *,
    enabled: bool,
    world_size: int,
    local_world_size: int,
    bsz: int,
    complex_bytes: int,
    max_bond: int | None,
    cutoff: float,
    backward_backend: str,
    distributed_profile: str | None,
    inspect_devices: bool,
    assume_devices_ready: bool,
) -> DistributedTrainingPreflight:
    """Collect optional planner evidence without importing JAX execution code."""

    if not enabled:
        return DistributedTrainingPreflight()

    statevector_summary: Mapping[str, Any] | None = None
    statevector_blockers: tuple[str, ...] = ()
    statevector_claim_allowed = False
    try:
        from ..runtime.planner_adapter import plan_jax_statevector_training

        training = plan_jax_statevector_training(
            ir,
            world_size=world_size,
            local_world_size=local_world_size,
            bsz=bsz,
            complex_bytes=complex_bytes,
            backward_backend=backward_backend,
            distributed_profile=distributed_profile,
            inspect_devices=inspect_devices,
            assume_devices_ready=assume_devices_ready,
        )
        statevector_summary = training.summary()
        statevector_blockers = tuple(
            str(item) for item in statevector_summary.get("blockers", ())
        )
        statevector_claim_allowed = bool(
            statevector_summary.get("scalability_claim_allowed", False)
        )
    except Exception as exc:
        statevector_summary = _unavailable_summary(exc)
        statevector_blockers = ("jax_sharded_statevector_training_plan_unavailable",)

    mps_summary: Mapping[str, Any] | None = None
    mps_blockers: tuple[str, ...] = ()
    try:
        from ..runtime.planner_adapter import plan_jax_mps_training

        training = plan_jax_mps_training(
            ir,
            world_size=world_size,
            local_world_size=local_world_size,
            bsz=bsz,
            complex_bytes=complex_bytes,
            max_bond=max_bond,
            cutoff=cutoff,
            backward_backend=backward_backend,
            distributed_profile=distributed_profile,
            inspect_devices=inspect_devices,
            assume_devices_ready=assume_devices_ready,
        )
        mps_summary = training.summary()
        mps_blockers = tuple(
            dict.fromkeys(
                (
                    *(str(item) for item in mps_summary.get("blockers", ())),
                    *(
                        str(item)
                        for item in mps_summary.get(
                            "mps_runtime_blockers",
                            mps_summary.get("phase5_mps_runtime_blockers", ()),
                        )
                    ),
                )
            )
        )
    except Exception as exc:
        mps_summary = _unavailable_summary(exc)
        mps_blockers = ("jax_sharded_mps_training_plan_unavailable",)

    return DistributedTrainingPreflight(
        statevector_summary=statevector_summary,
        statevector_blockers=statevector_blockers,
        statevector_claim_allowed=statevector_claim_allowed,
        mps_summary=mps_summary,
        mps_blockers=mps_blockers,
    )
