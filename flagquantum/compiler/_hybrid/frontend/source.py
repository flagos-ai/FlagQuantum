"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.frontend.source import (
    _function_node as _function_node,
)
from flagquantum.compiler.frontends._structured.frontend.source import (
    capture_function as capture_function,
)
from flagquantum.compiler.frontends._structured.frontend.source import (
    capture_source as capture_source,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.frontend.source"
)
_sys.modules[__name__] = _implementation
