"""Fail-closed identity records for distributed communication routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


class DistributedIdentityError(RuntimeError):
    """Raised when a requested distributed route lacks required evidence."""


def _frozen(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class DistributedIdentity:
    """Serializable process-group identity that never infers its inner route.

    ``backend="flagos"`` proves only that PyTorch attached to Torch-FL's public
    ProcessGroup boundary. It does not prove that the provider selected FlagCX,
    avoided host staging, or exercised a particular collective implementation.
    Those facts require later provider-owned conformance evidence.
    """

    outer_backend: str
    logical_device: str
    rank: int
    world_size: int
    process_group_initialized: bool
    provider: str
    provider_version: str | None = None
    inner_backend: str | None = None
    inner_backend_verified: bool = False
    flagcx_route_verified: bool = False
    host_staging_observed: bool | None = None
    communication_claim_allowed: bool = False
    blockers: tuple[str, ...] = ()
    platform_identity: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "flagquantum_distributed_identity_v1"

    def __post_init__(self) -> None:
        if not self.outer_backend or not self.logical_device or not self.provider:
            raise ValueError(
                "outer_backend, logical_device, and provider must be non-empty"
            )
        if self.rank < 0 or self.world_size < 1 or self.rank >= self.world_size:
            raise ValueError("rank must identify one member of world_size")
        if self.flagcx_route_verified and (
            self.outer_backend != "flagos"
            or self.inner_backend != "flagcx"
            or not self.inner_backend_verified
            or self.host_staging_observed is not False
        ):
            raise ValueError(
                "verified FlagCX requires flagos, a verified flagcx inner backend, "
                "and explicit no-host-staging evidence"
            )
        if self.communication_claim_allowed and self.blockers:
            raise ValueError("communication claims cannot retain identity blockers")
        if self.communication_claim_allowed and not self.flagcx_route_verified:
            raise ValueError("communication claims require a verified FlagCX route")
        object.__setattr__(self, "platform_identity", _frozen(self.platform_identity))

    @property
    def flagcx_route_status(self) -> str:
        if self.flagcx_route_verified:
            return "verified"
        if self.outer_backend == "flagos":
            return "unverified"
        return "not_applicable"

    def claim_blockers_for(self, backend: str) -> tuple[str, ...]:
        """Expose identity blockers only for the FlagOS evidence boundary."""

        return self.blockers if str(backend).strip().lower() == "flagos" else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "outer_backend": self.outer_backend,
            "logical_device": self.logical_device,
            "rank": self.rank,
            "world_size": self.world_size,
            "process_group_initialized": self.process_group_initialized,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "inner_backend": self.inner_backend,
            "inner_backend_verified": self.inner_backend_verified,
            "flagcx_route_verified": self.flagcx_route_verified,
            "flagcx_route_status": self.flagcx_route_status,
            "host_staging_observed": self.host_staging_observed,
            "communication_claim_allowed": self.communication_claim_allowed,
            "blockers": self.blockers,
            "platform_identity": dict(self.platform_identity),
        }


def build_distributed_identity(
    *,
    outer_backend: str,
    logical_device: str,
    rank: int,
    world_size: int,
    process_group_initialized: bool,
    platform_identity: Mapping[str, Any] | None = None,
) -> DistributedIdentity:
    """Build identity from observable public runtime facts only."""

    backend = str(outer_backend).strip().lower()
    device = str(logical_device)
    platform = dict(platform_identity or {})
    if backend == "flagos":
        provider = str(platform.get("provider") or "torch_fl")
        blockers = [
            "flagcx_inner_backend_unverified",
            "flagcx_collective_conformance_pending",
            "host_staging_unverified",
        ]
        if not process_group_initialized:
            blockers.insert(0, "flagos_process_group_not_initialized")
        return DistributedIdentity(
            outer_backend=backend,
            logical_device=device,
            rank=int(rank),
            world_size=int(world_size),
            process_group_initialized=bool(process_group_initialized),
            provider=provider,
            provider_version=platform.get("provider_version"),
            blockers=tuple(blockers),
            platform_identity=platform,
        )

    blockers = ("distributed_identity_is_not_collective_conformance_evidence",)
    return DistributedIdentity(
        outer_backend=backend,
        logical_device=device,
        rank=int(rank),
        world_size=int(world_size),
        process_group_initialized=bool(process_group_initialized),
        provider="pytorch",
        inner_backend=backend if process_group_initialized else None,
        inner_backend_verified=bool(process_group_initialized),
        host_staging_observed=None,
        blockers=blockers,
        platform_identity=platform,
    )


def require_verified_flagcx(identity: DistributedIdentity) -> None:
    """Reject an identity unless provider-owned evidence proves FlagCX."""

    if not identity.flagcx_route_verified or not identity.communication_claim_allowed:
        details = ", ".join(identity.blockers) or "communication claim is disabled"
        raise DistributedIdentityError(f"FlagCX route is not verified: {details}")


def backend_uses_accelerator_tensors(backend: str) -> bool:
    """Return whether collective payloads must remain on an accelerator."""

    return str(backend).strip().lower() in {"nccl", "flagos"}


__all__ = (
    "DistributedIdentity",
    "DistributedIdentityError",
    "backend_uses_accelerator_tensors",
    "build_distributed_identity",
    "require_verified_flagcx",
)
