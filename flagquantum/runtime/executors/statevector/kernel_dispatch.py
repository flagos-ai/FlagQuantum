"""Deterministic kernel selection policy for the statevector backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import metadata
from typing import Any

from ....kernels.catalog import KernelRequest, match_kernel_implementations
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
    compiler_identity_status: str | None = None
    integration_path: str | None = None
    semantic_id: str | None = None
    implementation_id: str | None = None
    catalog_mismatches: tuple[str, ...] = ()

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
                        "identity_status": self.compiler_identity_status,
                    }
                    if self.accelerated
                    else None
                ),
                "kernel_route": {
                    "semantic_id": self.semantic_id or f"statevector.{self.feature}",
                    "implementation": (
                        "triton" if self.accelerated else "pytorch_eager"
                    ),
                    "integration_path": self.integration_path,
                    "fallback": self.fallback,
                },
            }
        )
        if self.semantic_id is not None:
            summary["kernel_route"].update(
                {
                    "implementation_id": self.implementation_id,
                    "catalog_mismatches": self.catalog_mismatches,
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
        record_list: list[dict[str, Any]] = []
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
            record_list.append(record)
        records = tuple(record_list)
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


def triton_compiler_provenance() -> tuple[str | None, str | None, str, str]:
    """Resolve who installed the ``triton`` module without importing it."""

    module_owners = metadata.packages_distributions().get("triton", ())
    owners = tuple(
        sorted({owner.casefold().replace("_", "-") for owner in module_owners})
    )
    if not owners:
        return None, None, "unknown", "missing"
    if len(owners) != 1:
        return None, None, "unknown", "ambiguous"

    distribution = owners[0]
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return distribution, None, "unknown", "missing_metadata"

    if distribution == "triton":
        return distribution, version, "direct", "resolved"
    if distribution == "flagtree":
        return distribution, version, "flagtree", "resolved"
    return distribution, version, "unknown", "unsupported_distribution"


def select_triton_kernel(
    feature: str,
    *,
    requested: bool,
    supported: bool = True,
    available: bool | None = None,
    device_runtime_provider: str | None = None,
    device_type: str | None = None,
    compiler_backend: str | None = None,
    capture_compiler_identity: bool = False,
) -> KernelDecision:
    """Select Triton or the portable PyTorch implementation with a reason."""

    common_identity = {
        "device_runtime_provider": device_runtime_provider,
        "device_type": device_type,
        "compiler_backend": compiler_backend,
        "integration_path": "pytorch" if capture_compiler_identity else None,
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
    compiler_identity_status = None
    if capture_compiler_identity:
        (
            compiler_distribution,
            compiler_version,
            integration_path,
            compiler_identity_status,
        ) = triton_compiler_provenance()
        common_identity["integration_path"] = integration_path
    return KernelDecision(
        feature,
        "triton",
        "eligible",
        compiler_distribution=compiler_distribution,
        compiler_version=compiler_version,
        compiler_identity_status=compiler_identity_status,
        **common_identity,
    )


def select_cataloged_triton_kernel(
    feature: str,
    *,
    request: KernelRequest,
    implementation_id: str,
    requested: bool,
    runtime_supported: bool = True,
    available: bool | None = None,
    device_runtime_provider: str | None = None,
    compiler_backend: str | None = None,
    capture_compiler_identity: bool = False,
) -> KernelDecision:
    """Select one wired Triton implementation after catalog capability matching.

    The catalog establishes whether the named implementation declares and
    evidences the requested capabilities. Runtime policy, availability, and
    fallback precedence remain owned by :func:`select_triton_kernel`.
    """

    match = match_kernel_implementations(request)
    candidate_ids = {
        candidate.implementation.implementation_id for candidate in match.candidates
    }
    catalog_supported = implementation_id in candidate_ids
    mismatch_codes: tuple[str, ...] = ()
    if not catalog_supported:
        rejection = next(
            (
                rejection
                for rejection in match.rejections
                if rejection.implementation.implementation_id == implementation_id
            ),
            None,
        )
        if rejection is not None:
            mismatch_codes = tuple(mismatch.code for mismatch in rejection.mismatches)
        elif not match.semantic_known:
            mismatch_codes = ("semantic_unknown",)
        else:
            mismatch_codes = ("implementation_not_registered_for_semantic",)

    decision = select_triton_kernel(
        feature,
        requested=requested,
        supported=runtime_supported and catalog_supported,
        available=available,
        device_runtime_provider=device_runtime_provider,
        device_type=request.device,
        compiler_backend=compiler_backend,
        capture_compiler_identity=capture_compiler_identity,
    )
    return replace(
        decision,
        semantic_id=request.semantic_id,
        implementation_id=implementation_id if decision.accelerated else None,
        catalog_mismatches=mismatch_codes,
    )


__all__ = (
    "KernelDecision",
    "KernelDispatchEvidence",
    "select_cataloged_triton_kernel",
    "select_triton_kernel",
    "triton_available",
    "triton_compiler_provenance",
)
