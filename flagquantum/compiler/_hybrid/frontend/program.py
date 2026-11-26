"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.frontend.program import (
    _Capture as _Capture,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.frontend.program"
)
_sys.modules[__name__] = _implementation
