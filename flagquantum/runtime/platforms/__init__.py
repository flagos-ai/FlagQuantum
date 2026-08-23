"""Device-platform boundaries for FlagQuantum runtimes.

Platform modules own device discovery and device-specific lifecycle operations.
Representation runtimes should depend on this package instead of importing a
vendor runtime directly.
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
