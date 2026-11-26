"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.dynamics.adjoint import __all__ as __all__
from flagquantum.simulation.dynamics.adjoint import (
    _adjoint_gradient as _adjoint_gradient,
)
from flagquantum.simulation.dynamics.adjoint import (
    adjoint_gradient as adjoint_gradient,
)

_implementation = _import_module("flagquantum.simulation.dynamics.adjoint")
_sys.modules[__name__] = _implementation
