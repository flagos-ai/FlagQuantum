"""PennyLane Lightning provider for the bounded research VQE capability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata, util

import torch

from flagquantum.circuit import Circuit
from flagquantum.ecosystem.research_vqe_provider import (
    ResearchVQEProviderIdentity,
    run_statevector_research_vqe,
)
from flagquantum.services.ground_state import (
    GroundStateVQERequest,
    GroundStateVQEResult,
)
from flagquantum.services.research_providers import ResearchProviderAvailability

_PROVIDER_ID = "pennylane-lightning"
_ADAPTER_VERSION = "1.0.0"
_TOOL_NAME = "flagquantum.ecosystem.pennylane.research_provider.run_vqe"

StatevectorRunner = Callable[[Circuit], torch.Tensor]


def _run_statevector(circuit: Circuit) -> torch.Tensor:
    from .lightning import run

    result = run(circuit)
    if result.state is None:
        raise RuntimeError(
            "PennyLane Lightning did not return the requested statevector"
        )
    return result.state


def _dependency_reason() -> str | None:
    missing = tuple(
        name
        for name in ("pennylane", "pennylane_lightning")
        if util.find_spec(name) is None
    )
    if missing:
        return "missing optional dependency: " + ", ".join(missing)
    try:
        metadata.version("pennylane-lightning")
    except metadata.PackageNotFoundError:
        return "missing optional distribution: pennylane-lightning"
    return None


def _package_version() -> str:
    try:
        return metadata.version("pennylane-lightning")
    except metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True)
class PennyLaneLightningResearchVQEProvider:
    """Execute the P0 ansatz through the explicit Lightning bridge."""

    provider_id: str = _PROVIDER_ID
    adapter_version: str = _ADAPTER_VERSION
    _runner: StatevectorRunner = field(
        default=_run_statevector, repr=False, compare=False
    )
    _probe: Callable[[], str | None] = field(
        default=_dependency_reason, repr=False, compare=False
    )

    def availability(self) -> ResearchProviderAvailability:
        reason = self._probe()
        return ResearchProviderAvailability(self.provider_id, reason is None, reason)

    def estimate_backend_executions(self, request: GroundStateVQERequest) -> int:
        return 3 * (request.optimizer.max_evaluations - 1) + 1

    def run_vqe(self, request: GroundStateVQERequest) -> GroundStateVQEResult:
        return run_statevector_research_vqe(
            request,
            runner=self._runner,
            identity=ResearchVQEProviderIdentity(
                provider="PennyLane Lightning",
                package="pennylane-lightning",
                package_version=_package_version(),
                tool=_TOOL_NAME,
                tool_version=self.adapter_version,
            ),
        )


def create_research_vqe_provider() -> PennyLaneLightningResearchVQEProvider:
    """Create the optional provider without importing PennyLane."""

    return PennyLaneLightningResearchVQEProvider()


__all__ = (
    "PennyLaneLightningResearchVQEProvider",
    "create_research_vqe_provider",
)
