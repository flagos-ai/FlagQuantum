"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.provenance import __all__ as __all__
from flagquantum.simulation.kernels.provenance import (
    triton_compiler_provenance as triton_compiler_provenance,
)

_implementation = _import_module("flagquantum.simulation.kernels.provenance")
_sys.modules[__name__] = _implementation
