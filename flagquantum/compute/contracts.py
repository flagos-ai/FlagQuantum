"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.contracts import (
    MemorySnapshot as MemorySnapshot,
)
from flagquantum.compute.local.contracts import (
    PlatformActivationError as PlatformActivationError,
)
from flagquantum.compute.local.contracts import (
    PlatformDevice as PlatformDevice,
)
from flagquantum.compute.local.contracts import (
    PlatformError as PlatformError,
)
from flagquantum.compute.local.contracts import (
    PlatformIdentity as PlatformIdentity,
)
from flagquantum.compute.local.contracts import (
    PlatformRuntime as PlatformRuntime,
)
from flagquantum.compute.local.contracts import (
    PlatformUnavailableError as PlatformUnavailableError,
)
from flagquantum.compute.local.contracts import __all__ as __all__
from flagquantum.compute.local.contracts import (
    _frozen_metadata as _frozen_metadata,
)

_implementation = _import_module("flagquantum.compute.local.contracts")
_sys.modules[__name__] = _implementation
