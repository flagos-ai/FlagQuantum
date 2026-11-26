"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton._jit import (
    Kernel as Kernel,
)
from flagquantum.simulation.kernels.triton._jit import __all__ as __all__
from flagquantum.simulation.kernels.triton._jit import (
    autotune as autotune,
)
from flagquantum.simulation.kernels.triton._jit import (
    jit as jit,
)

_implementation = _import_module("flagquantum.simulation.kernels.triton._jit")
_sys.modules[__name__] = _implementation
