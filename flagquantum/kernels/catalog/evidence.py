"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.catalog.evidence import (
    EVIDENCE as EVIDENCE,
)
from flagquantum.simulation.kernels.catalog.evidence import __all__ as __all__
from flagquantum.simulation.kernels.catalog.evidence import (
    _evidence as _evidence,
)

_implementation = _import_module("flagquantum.simulation.kernels.catalog.evidence")
_sys.modules[__name__] = _implementation
