"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.results.adapters import (
    LiveRuntimeSummary as LiveRuntimeSummary,
)
from flagquantum.runtime.results.adapters import __all__ as __all__
from flagquantum.runtime.results.adapters import (
    _module_execution_result as _module_execution_result,
)

_implementation = _import_module("flagquantum.runtime.results.adapters")
_sys.modules[__name__] = _implementation
