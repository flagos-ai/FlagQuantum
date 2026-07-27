"""Typed adapter from planner objects to cross-layer plan contracts."""

from __future__ import annotations

import hashlib
from typing import Any

from ..core.contracts import (
    CapabilityContract,
    EstimatedResources,
    RequestedExecution,
    RuntimePlanContract,
)


def execution_plan_contract(plan: Any) -> RuntimePlanContract:
    config = dict(plan.runtime_config or {})
    requested = RequestedExecution(
        mode=plan.state_mode,
        backend=str(config.get("backend", "pytorch")),
        device=str(config.get("device", "cpu")),
        dtype=str(config.get("complex_dtype", "complex64")),
        world_size=plan.world_size,
    )
    identity = (
        f"{plan.analysis.n_wires}:{plan.analysis.n_instructions}:"
        f"{plan.state_mode}:{plan.world_size}:{plan.state_bytes}"
    )
    return RuntimePlanContract(
        requested=requested,
        estimated=EstimatedResources(
            memory_bytes=plan.state_bytes,
            depth=plan.analysis.depth,
        ),
        capability=CapabilityContract(
            backend=requested.backend,
            devices=(requested.device,),
            dtypes=(requested.dtype,),
            modes=(plan.state_mode,),
            supports_distributed=plan.world_size > 1,
        ),
        plan_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
    )


__all__ = ["execution_plan_contract"]
