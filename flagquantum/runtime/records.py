"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.results.records import __all__ as __all__
from flagquantum.runtime.results.records import (
    record_execution as record_execution,
)

_implementation = _import_module("flagquantum.runtime.results.records")
_sys.modules[__name__] = _implementation
