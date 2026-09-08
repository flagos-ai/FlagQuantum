"""Memory-aware checkpoint policy for statevector reverse execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from ....providers.platform import get_platform_runtime


@dataclass(frozen=True)
class StatevectorCheckpointPolicy:
    """Versioned rematerialization policy for deep sharded circuits."""

    version: str = "statevector_checkpoint_v2"
    strategy: str = "auto"
    interval: int = 0
    memory_budget_bytes: int | None = None
    selection_reason: str = "unresolved"
    estimated_local_state_bytes: int = 0
    estimated_required_bytes: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in {
            "auto",
            "full_rematerialization",
            "interval",
            "reversible_adjoint",
        }:
            raise ValueError(f"unknown checkpoint strategy {self.strategy!r}")
        if self.strategy == "interval" and self.interval <= 0:
            raise ValueError("interval checkpoint strategy requires interval > 0")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("checkpoint memory_budget_bytes must be positive")


def _checkpoint_memory_budget(
    policy: StatevectorCheckpointPolicy, *, device: Any
) -> int:
    if policy.memory_budget_bytes is not None:
        return int(policy.memory_budget_bytes)
    configured = os.getenv("FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES")
    if configured is not None:
        budget = int(configured)
        if budget <= 0:
            raise ValueError("FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES must be positive")
        return budget
    if device.type == "cuda":
        platform = get_platform_runtime(device.type)
        if platform.is_available():
            free_bytes = platform.memory_snapshot(device).free_bytes
            if free_bytes is not None:
                return int(free_bytes * 0.7)
    return 512 << 20


def resolve_checkpoint_policy(
    policy: StatevectorCheckpointPolicy,
    *,
    local_state_bytes: int,
    device: Any,
) -> StatevectorCheckpointPolicy:
    """Resolve ``auto`` using a conservative rank-local peak-memory model."""

    state_bytes = int(local_state_bytes)
    if policy.strategy != "auto":
        return replace(
            policy,
            selection_reason="explicit_strategy",
            estimated_local_state_bytes=state_bytes,
            estimated_required_bytes=(
                state_bytes * 4
                if policy.strategy == "reversible_adjoint"
                else state_bytes * 3
            ),
        )
    budget = _checkpoint_memory_budget(policy, device=device)
    reversible_required = state_bytes * 4
    if reversible_required <= budget:
        return replace(
            policy,
            strategy="reversible_adjoint",
            memory_budget_bytes=budget,
            selection_reason="forward_state_reuse_fits_budget",
            estimated_local_state_bytes=state_bytes,
            estimated_required_bytes=reversible_required,
        )
    return replace(
        policy,
        strategy="full_rematerialization",
        memory_budget_bytes=budget,
        selection_reason="reversible_working_set_exceeds_budget",
        estimated_local_state_bytes=state_bytes,
        estimated_required_bytes=reversible_required,
    )


__all__ = ("StatevectorCheckpointPolicy", "resolve_checkpoint_policy")
