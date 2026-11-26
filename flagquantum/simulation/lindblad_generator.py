"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.dynamics.generator import (
    DEFAULT_DENSE_GENERATOR_BYTES as DEFAULT_DENSE_GENERATOR_BYTES,
)
from flagquantum.simulation.dynamics.generator import (
    Liouvillian as Liouvillian,
)
from flagquantum.simulation.dynamics.generator import __all__ as __all__

_implementation = _import_module("flagquantum.simulation.dynamics.generator")
_sys.modules[__name__] = _implementation
