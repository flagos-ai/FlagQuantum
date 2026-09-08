"""Small, vendor-neutral contracts for compute-platform integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import torch


class PlatformError(RuntimeError):
    """Base class for platform discovery and activation failures."""


class PlatformActivationError(PlatformError):
    """An optional compute platform could not be activated safely."""


class PlatformUnavailableError(PlatformError):
    """A requested platform is installed but unavailable in this process."""


def _frozen_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class PlatformDevice:
    """A concrete device discovered through one compute runtime."""

    device_type: str
    index: int | None
    name: str
    provider: str
    available: bool = True
    memory_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device_type or not self.provider:
            raise ValueError("device_type and provider must be non-empty")
        if self.index is not None and self.index < 0:
            raise ValueError("device index must be non-negative")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @property
    def device(self) -> torch.device:
        if self.index is None:
            return torch.device(self.device_type)
        return torch.device(self.device_type, self.index)


@dataclass(frozen=True)
class MemorySnapshot:
    """Portable memory facts; unavailable values remain explicit."""

    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None


@dataclass(frozen=True)
class PlatformIdentity:
    """Serializable identity used for diagnostics, never capability promotion."""

    provider: str
    device_type: str
    torch_version: str
    provider_version: str | None = None
    vendor: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "device_type": self.device_type,
            "torch_version": self.torch_version,
            "provider_version": self.provider_version,
            "vendor": self.vendor,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class PlatformRuntime(Protocol):
    """Minimum lifecycle surface implemented by directly controlled devices."""

    name: str
    device_type: str
    optional_dependency: str | None

    def installed(self) -> bool: ...

    def activated(self) -> bool: ...

    def activate(self) -> None: ...

    def is_available(self) -> bool: ...

    def discover(self) -> tuple[PlatformDevice, ...]: ...

    def devices(self) -> tuple[PlatformDevice, ...]: ...

    def synchronize(self, device: torch.device) -> None: ...

    def memory_snapshot(self, device: torch.device) -> MemorySnapshot: ...

    def stream(self, device: torch.device, priority: int = 0) -> Any: ...

    def event(self, device: torch.device) -> Any: ...

    def rng_state(self, device: torch.device) -> torch.Tensor: ...

    def restore_rng_state(self, device: torch.device, state: torch.Tensor) -> None: ...

    def profiler_metadata(self, device: torch.device) -> Mapping[str, Any]: ...

    def identity(self) -> PlatformIdentity: ...


__all__ = (
    "MemorySnapshot",
    "PlatformActivationError",
    "PlatformDevice",
    "PlatformError",
    "PlatformIdentity",
    "PlatformRuntime",
    "PlatformUnavailableError",
)
