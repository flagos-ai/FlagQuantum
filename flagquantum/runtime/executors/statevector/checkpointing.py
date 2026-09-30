"""Memory-aware checkpoint policy for statevector reverse execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from math import ceil
from typing import Any

from ....compute import get_platform_runtime


@dataclass(frozen=True)
class StatevectorCheckpointPolicy:
    """Versioned rematerialization policy for deep sharded circuits."""

    version: str = "statevector_checkpoint_v3"
    strategy: str = "auto"
    interval: int = 0
    memory_budget_bytes: int | None = None
    selection_reason: str = "unresolved"
    estimated_local_state_bytes: int = 0
    estimated_required_bytes: int = 0
    estimated_reversible_bytes: int = 0
    estimated_checkpoint_count: int = 0

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
    platform = get_platform_runtime(device.type)
    if platform.is_available():
        free_bytes = platform.memory_snapshot(device).free_bytes
        if free_bytes is not None:
            # Keep enough headroom for Python, Torch's allocator, shared
            # libraries, and concurrent host activity. The resulting budget is
            # a planner input, not a process-RSS cap or allocation promise.
            return max(1, int(free_bytes * 0.7))
    return 512 << 20


def resolve_checkpoint_policy(
    policy: StatevectorCheckpointPolicy,
    *,
    local_state_bytes: int,
    device: Any,
    instruction_count: int = 0,
    contains_local_cx: bool = False,
) -> StatevectorCheckpointPolicy:
    """Resolve ``auto`` using a conservative rank-local peak-memory model."""

    state_bytes = int(local_state_bytes)
    if state_bytes <= 0:
        raise ValueError("local_state_bytes must be positive")
    if instruction_count < 0:
        raise ValueError("instruction_count must be non-negative")
    reversible_states = 5 if device.type == "cpu" and contains_local_cx else 4
    reversible_required = state_bytes * reversible_states
    if policy.strategy != "auto":
        checkpoint_count = (
            (instruction_count - 1) // policy.interval + 1
            if policy.strategy == "interval" and instruction_count > 0
            else 0
        )
        if policy.strategy == "reversible_adjoint":
            required = reversible_required
        elif policy.strategy == "interval":
            required = state_bytes * (checkpoint_count + 2)
        else:
            required = state_bytes * 3
        return replace(
            policy,
            selection_reason="explicit_strategy",
            estimated_local_state_bytes=state_bytes,
            estimated_required_bytes=required,
            estimated_reversible_bytes=reversible_required,
            estimated_checkpoint_count=checkpoint_count,
        )
    budget = _checkpoint_memory_budget(policy, device=device)
    if reversible_required <= budget:
        return replace(
            policy,
            strategy="reversible_adjoint",
            memory_budget_bytes=budget,
            selection_reason="forward_state_reuse_fits_budget",
            estimated_local_state_bytes=state_bytes,
            estimated_required_bytes=reversible_required,
            estimated_reversible_bytes=reversible_required,
        )
    # An interval sweep retains its block-boundary checkpoints plus one adjoint
    # and one rematerialized ket. The initial state is itself a checkpoint.
    checkpoint_capacity = budget // state_bytes - 2
    if instruction_count > 1 and checkpoint_capacity >= 2:
        interval = ceil(instruction_count / checkpoint_capacity)
        checkpoint_count = (instruction_count - 1) // interval + 1
        required = state_bytes * (checkpoint_count + 2)
        return replace(
            policy,
            strategy="interval",
            interval=interval,
            memory_budget_bytes=budget,
            selection_reason="budgeted_block_checkpoints",
            estimated_local_state_bytes=state_bytes,
            estimated_required_bytes=required,
            estimated_reversible_bytes=reversible_required,
            estimated_checkpoint_count=checkpoint_count,
        )
    return replace(
        policy,
        strategy="full_rematerialization",
        memory_budget_bytes=budget,
        selection_reason="checkpoint_working_set_exceeds_budget",
        estimated_local_state_bytes=state_bytes,
        estimated_required_bytes=state_bytes * 3,
        estimated_reversible_bytes=reversible_required,
        estimated_checkpoint_count=1,
    )


__all__ = ("StatevectorCheckpointPolicy", "resolve_checkpoint_policy")
