"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.catalog.semantics import (
    SEMANTICS as SEMANTICS,
)
from flagquantum.simulation.kernels.catalog.semantics import __all__ as __all__
from flagquantum.simulation.kernels.catalog.semantics import (
    _semantic as _semantic,
)

_implementation = _import_module("flagquantum.simulation.kernels.catalog.semantics")
_sys.modules[__name__] = _implementation
