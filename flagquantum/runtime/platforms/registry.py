"""Task-local-free registry for built-in and optional platform providers."""

from __future__ import annotations

from typing import Mapping

import torch

from .contracts import PlatformDevice, PlatformRuntime, PlatformUnavailableError
from .flagos import FlagOSPlatformRuntime
from .pytorch import CPUPlatformRuntime, CUDAPlatformRuntime

_PLATFORMS: Mapping[str, PlatformRuntime] = {
    "cpu": CPUPlatformRuntime(),
    "cuda": CUDAPlatformRuntime(),
    "flagos": FlagOSPlatformRuntime(),
}


def get_platform_runtime(device_type: str) -> PlatformRuntime:
    """Return a provider without activating optional dependencies."""

    normalized = str(device_type).lower().split(":", 1)[0]
    try:
        return _PLATFORMS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown FlagQuantum platform {device_type!r}.") from exc


def list_platform_status() -> tuple[dict[str, object], ...]:
    """Report install/activation state without importing optional providers."""

    return tuple(
        {
            "device_type": name,
            "provider": platform.name,
            "optional_dependency": platform.optional_dependency,
            "installed": platform.installed(),
            "activated": platform.activated(),
            "available": (
                platform.is_available()
                if platform.optional_dependency is None or platform.activated()
                else None
            ),
        }
        for name, platform in _PLATFORMS.items()
    )


def discover_platform_devices() -> tuple[PlatformDevice, ...]:
    """Discover devices without importing an inactive optional provider."""

    devices: list[PlatformDevice] = []
    for platform in _PLATFORMS.values():
        if platform.optional_dependency is not None and not platform.activated():
            continue
        devices.extend(platform.discover())
    return tuple(devices)


def resolve_platform_device(
    device: str | torch.device,
    *,
    activate_optional: bool = True,
) -> torch.device:
    """Resolve a concrete device and fail before workload allocation."""

    requested = str(device)
    device_type = requested.lower().split(":", 1)[0]
    platform = get_platform_runtime(device_type)
    if platform.optional_dependency is not None and not platform.activated():
        if not activate_optional:
            raise PlatformUnavailableError(
                f"Optional platform {device_type!r} has not been activated."
            )
        platform.activate()
    if not platform.is_available():
        raise PlatformUnavailableError(
            f"Platform {device_type!r} is not available in this process."
        )
    resolved = torch.device(device)
    if resolved.index is not None:
        available_indices = {
            item.index for item in platform.discover() if item.available
        }
        if resolved.index not in available_indices:
            raise PlatformUnavailableError(
                f"Platform {device_type!r} does not expose device index "
                f"{resolved.index}."
            )
    return resolved


__all__ = (
    "discover_platform_devices",
    "get_platform_runtime",
    "list_platform_status",
    "resolve_platform_device",
)
