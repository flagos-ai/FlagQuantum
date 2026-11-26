"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.flagos import (
    FlagOSPlatformRuntime as FlagOSPlatformRuntime,
)
from flagquantum.compute.local.flagos import __all__ as __all__

_implementation = _import_module("flagquantum.compute.local.flagos")
_sys.modules[__name__] = _implementation
