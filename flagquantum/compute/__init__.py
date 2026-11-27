"""Local resources, external compute services, and realtime protocols.

Direct-device exports retain their historical public import paths and load only
when requested, so importing a protocol does not activate Torch or a provider.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .local import (
        MemorySnapshot as MemorySnapshot,
    )
    from .local import (
        PlatformActivationError as PlatformActivationError,
    )
    from .local import (
        PlatformDevice as PlatformDevice,
    )
    from .local import (
        PlatformIdentity as PlatformIdentity,
    )
    from .local import (
        PlatformRuntime as PlatformRuntime,
    )
    from .local import (
        PlatformUnavailableError as PlatformUnavailableError,
    )
    from .local import (
        discover_platform_devices as discover_platform_devices,
    )
    from .local import (
        get_platform_runtime as get_platform_runtime,
    )
    from .local import (
        list_platform_status as list_platform_status,
    )
    from .local import (
        resolve_platform_device as resolve_platform_device,
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


def __getattr__(name: str) -> Any:
    if name in {"local", "remote", "realtime"}:
        return import_module(f".{name}", __name__)
    if name in __all__:
        value = getattr(import_module(".local", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
