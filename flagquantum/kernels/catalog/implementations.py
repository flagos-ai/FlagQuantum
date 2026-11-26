"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.catalog.implementations import (
    IMPLEMENTATIONS as IMPLEMENTATIONS,
)
from flagquantum.simulation.kernels.catalog.implementations import __all__ as __all__
from flagquantum.simulation.kernels.catalog.implementations import (
    _flagtree as _flagtree,
)
from flagquantum.simulation.kernels.catalog.implementations import (
    _triton as _triton,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.catalog.implementations"
)
_sys.modules[__name__] = _implementation
