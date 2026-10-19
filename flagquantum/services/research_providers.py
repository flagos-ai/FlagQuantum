"""Internal provider seam for bounded research capability execution.

Provider objects are process-local dependencies, not a public mutable registry.
The public contract names providers through manifests and execution policy;
adapters remain owned by their scientific or ecosystem domains.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from .ground_state import (
    GroundStateVQERequest,
    GroundStateVQEResult,
    run_ground_state_vqe,
)


@dataclass(frozen=True, slots=True)
class ResearchProviderAvailability:
    """Deterministic local preflight for one installed provider adapter."""

    provider: str
    available: bool
    reason: str | None = None


class ResearchVQEProvider(Protocol):
    """Narrow provider contract for the P0 VQE operation."""

    @property
    def provider_id(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    def availability(self) -> ResearchProviderAvailability: ...

    def estimate_backend_executions(self, request: GroundStateVQERequest) -> int: ...

    def run_vqe(self, request: GroundStateVQERequest) -> GroundStateVQEResult: ...


@dataclass(frozen=True, slots=True)
class NativeResearchVQEProvider:
    """Built-in FlagQuantum implementation of the P0 VQE provider contract."""

    provider_id: str = "flagquantum-native"
    adapter_version: str = "1.0.0"

    def availability(self) -> ResearchProviderAvailability:
        return ResearchProviderAvailability(self.provider_id, True)

    def estimate_backend_executions(self, request: GroundStateVQERequest) -> int:
        return request.optimizer.max_evaluations

    def run_vqe(self, request: GroundStateVQERequest) -> GroundStateVQEResult:
        return run_ground_state_vqe(request)


def installed_vqe_providers() -> tuple[ResearchVQEProvider, ...]:
    """Return immutable installed adapters in deterministic default order."""

    native: ResearchVQEProvider = NativeResearchVQEProvider()
    qiskit_module = import_module("flagquantum.ecosystem.qiskit.research_provider")
    pennylane_module = import_module(
        "flagquantum.ecosystem.pennylane.research_provider"
    )
    qiskit_aer: ResearchVQEProvider = qiskit_module.create_research_vqe_provider()
    pennylane: ResearchVQEProvider = pennylane_module.create_research_vqe_provider()
    return (native, qiskit_aer, pennylane)


def plan_vqe_provider_route(
    request: GroundStateVQERequest,
    provider_order: tuple[str, ...],
    providers: tuple[ResearchVQEProvider, ...],
) -> tuple[dict[str, ResearchVQEProvider], dict[str, Any]]:
    """Assess every requested adapter without changing caller-owned order."""

    by_id = {provider.provider_id: provider for provider in providers}
    if len(by_id) != len(providers):
        raise RuntimeError("installed research provider IDs must be unique")
    candidates: list[dict[str, Any]] = []
    selected_provider: str | None = None
    for rank, provider_id in enumerate(provider_order, start=1):
        provider = by_id.get(provider_id)
        available = False
        reason: str | None = "provider adapter is not installed"
        error_code = "PROVIDER_NOT_INSTALLED"
        adapter_version: str | None = None
        estimated_backend_executions: int | None = None
        if provider is not None:
            adapter_version = provider.adapter_version
            estimated_backend_executions = provider.estimate_backend_executions(request)
            try:
                availability = provider.availability()
                if availability.provider != provider_id:
                    reason = "provider availability identity mismatch"
                    error_code = "PROVIDER_PREFLIGHT_FAILED"
                elif availability.available:
                    available = True
                    reason = None
                    error_code = ""
                else:
                    reason = availability.reason or "provider unavailable"
                    error_code = "PROVIDER_UNAVAILABLE"
            except (RuntimeError, ValueError) as exc:
                reason = str(exc)
                error_code = "PROVIDER_PREFLIGHT_FAILED"
        selected = available and selected_provider is None
        if selected:
            selected_provider = provider_id
        candidates.append(
            {
                "provider": provider_id,
                "rank": rank,
                "installed": provider is not None,
                "available": available,
                "score": len(provider_order) - rank + 1 if available else 0,
                "selected": selected,
                "reason": reason,
                "preflight_error_code": error_code or None,
                "adapter_version": adapter_version,
                "estimated_backend_executions": estimated_backend_executions,
            }
        )
    return by_id, {
        "selection_policy": "first_available_in_explicit_order",
        "requested_order": list(provider_order),
        "selected_provider": selected_provider,
        "candidates": candidates,
    }


def denied_vqe_provider_route(provider_order: tuple[str, ...]) -> dict[str, Any]:
    """Explain why a multi-provider route was not assessed."""

    return {
        "selection_policy": "first_available_in_explicit_order",
        "requested_order": list(provider_order),
        "selected_provider": None,
        "candidates": [],
        "decision": "fallback_authorization_required",
    }


def research_vqe_metrics(
    estimate: Any,
    route_plan: Mapping[str, Any],
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the content-hashed route explanation to VQE metrics."""

    return {
        "resource_estimate": estimate.to_dict(),
        "provider_route": dict(route_plan),
        **({} if values is None else values),
    }


__all__ = (
    "NativeResearchVQEProvider",
    "ResearchProviderAvailability",
    "ResearchVQEProvider",
    "denied_vqe_provider_route",
    "installed_vqe_providers",
    "plan_vqe_provider_route",
    "research_vqe_metrics",
)
