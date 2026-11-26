"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.registry import (
    _PLATFORMS as _PLATFORMS,
)
from flagquantum.compute.local.registry import __all__ as __all__
from flagquantum.compute.local.registry import (
    discover_platform_devices as discover_platform_devices,
)
from flagquantum.compute.local.registry import (
    get_platform_runtime as get_platform_runtime,
)
from flagquantum.compute.local.registry import (
    list_platform_status as list_platform_status,
)
from flagquantum.compute.local.registry import (
    resolve_platform_device as resolve_platform_device,
)

_implementation = _import_module("flagquantum.compute.local.registry")
_sys.modules[__name__] = _implementation
