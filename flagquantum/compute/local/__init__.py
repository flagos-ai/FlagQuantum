"""Compute resources controlled directly by the current process.

This package owns device discovery and device-specific lifecycle operations.
Runtime and Simulation depend on it instead of importing vendor runtimes
directly.
"""

from .contracts import (
    MemorySnapshot,
    PlatformActivationError,
    PlatformDevice,
    PlatformIdentity,
    PlatformRuntime,
    PlatformUnavailableError,
)
from .registry import (
    discover_platform_devices,
    get_platform_runtime,
    list_platform_status,
    resolve_platform_device,
)

__all__ = (
    "MemorySnapshot",
    "PlatformActivationError",
    "PlatformDevice",
    "PlatformIdentity",
    "PlatformRuntime",
    "PlatformUnavailableError",
    "discover_platform_devices",
    "get_platform_runtime",
    "list_platform_status",
    "resolve_platform_device",
)
