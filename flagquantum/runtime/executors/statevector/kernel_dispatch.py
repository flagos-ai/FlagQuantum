"""Deterministic kernel selection policy for the statevector backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from typing import Any

from .environment import mode


@dataclass(frozen=True)
class KernelDecision:
    """One auditable accelerated-kernel selection decision."""

    feature: str
    selected: str
    reason: str
    device_runtime_provider: str | None = None
    device_type: str | None = None
    compiler_distribution: str | None = None
    compiler_version: str | None = None
    compiler_backend: str | None = None
    integration_path: str | None = None

    @property
    def accelerated(self) -> bool:
        return self.selected == "triton"

    @property
    def fallback(self) -> bool:
        """Return whether acceleration was requested but could not be selected."""

        return self.selected == "pytorch" and self.reason in {
            "input_not_supported",
            "triton_unavailable",
        }

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "feature": self.feature,
            "selected": self.selected,
            "accelerated": self.accelerated,
            "reason": self.reason,
        }
        if self.integration_path is None:
            return summary

        summary.update(
            {
                "device_runtime": {
                    "provider": self.device_runtime_provider,
                    "device_type": self.device_type,
                },
                "kernel_compiler": (
                    {
                        "distribution": self.compiler_distribution,
                        "version": self.compiler_version,
                        "backend": self.compiler_backend,
                        "identity_source": "python_package_metadata",
                    }
                    if self.accelerated
                    else None
                ),
                "kernel_route": {
                    "semantic_id": f"statevector.{self.feature}",
                    "implementation": (
                        "triton" if self.accelerated else "pytorch_eager"
                    ),
                    "integration_path": self.integration_path,
                    "fallback": self.fallback,
                },
            }
        )
        return summary


@dataclass
class KernelDispatchEvidence:
    """Aggregated decisions made by kernels during one execution."""

    decisions: dict[KernelDecision, int] = field(default_factory=dict)

    def record(self, decision: KernelDecision, *, count: int = 1) -> None:
        self.decisions[decision] = self.decisions.get(decision, 0) + int(count)

    def summary(self) -> dict[str, Any]:
        records = []
        for decision, count in sorted(
            self.decisions.items(),
            key=lambda item: (
                item[0].feature,
                item[0].selected,
                item[0].reason,
                item[0].device_type or "",
            ),
        ):
            record = decision.summary()
            record.pop("accelerated")
            record["count"] = count
            records.append(record)
        records = tuple(records)
        return {
            "decisions": records,
            "triton_execution_count": sum(
                record["count"] for record in records if record["selected"] == "triton"
            ),
            "pytorch_fallback_count": sum(
                record["count"] for record in records if record["selected"] == "pytorch"
            ),
        }


@lru_cache(maxsize=1)
def triton_available() -> bool:
    """Return whether the optional Triton runtime is importable."""

    return importlib.util.find_spec("triton") is not None


def triton_distribution_identity() -> tuple[str | None, str | None]:
    """Read the installed Triton distribution identity without importing Triton."""

    try:
        distribution = metadata.distribution("triton")
    except metadata.PackageNotFoundError:
        return None, None
    return distribution.metadata.get("Name"), distribution.version


def select_triton_kernel(
    feature: str,
    *,
    requested: bool,
    supported: bool = True,
    available: bool | None = None,
    device_runtime_provider: str | None = None,
    device_type: str | None = None,
    compiler_backend: str | None = None,
    integration_path: str | None = None,
) -> KernelDecision:
    """Select Triton or the portable PyTorch implementation with a reason."""

    common_identity = {
        "device_runtime_provider": device_runtime_provider,
        "device_type": device_type,
        "compiler_backend": compiler_backend,
        "integration_path": integration_path,
    }
    if mode() == "portable":
        return KernelDecision(feature, "pytorch", "portable_mode", **common_identity)
    if not requested:
        return KernelDecision(
            feature, "pytorch", "disabled_by_policy", **common_identity
        )
    if not supported:
        return KernelDecision(
            feature, "pytorch", "input_not_supported", **common_identity
        )
    resolved_available = triton_available() if available is None else bool(available)
    if not resolved_available:
        return KernelDecision(
            feature, "pytorch", "triton_unavailable", **common_identity
        )
    compiler_distribution = None
    compiler_version = None
    if integration_path is not None:
        compiler_distribution, compiler_version = triton_distribution_identity()
    return KernelDecision(
        feature,
        "triton",
        "eligible",
        compiler_distribution=compiler_distribution,
        compiler_version=compiler_version,
        **common_identity,
    )


__all__ = (
    "KernelDecision",
    "KernelDispatchEvidence",
    "select_triton_kernel",
    "triton_available",
    "triton_distribution_identity",
)
