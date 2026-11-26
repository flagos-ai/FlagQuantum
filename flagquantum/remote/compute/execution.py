"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute.execution import (
    _CLIENTS as _CLIENTS,
)
from flagquantum.compute.remote.compute.execution import __all__ as __all__
from flagquantum.compute.remote.compute.execution import (
    execute_jiuding as execute_jiuding,
)

_implementation = _import_module("flagquantum.compute.remote.compute.execution")
_sys.modules[__name__] = _implementation
