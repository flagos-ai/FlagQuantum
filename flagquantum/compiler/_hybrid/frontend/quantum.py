"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.frontend.quantum import (
    capture_quantum_call as capture_quantum_call,
)
from flagquantum.compiler.frontends._structured.frontend.quantum import (
    capture_return as capture_return,
)
from flagquantum.compiler.frontends._structured.frontend.quantum import (
    parse_observable as parse_observable,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.frontend.quantum"
)
_sys.modules[__name__] = _implementation
