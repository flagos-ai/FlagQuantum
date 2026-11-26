"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_cx_fp64 import __all__ as __all__
from flagquantum.simulation.kernels.triton.statevector_cx_fp64 import (
    _cx as _cx,
)
from flagquantum.simulation.kernels.triton.statevector_cx_fp64 import (
    apply_cx as apply_cx,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_cx_fp64"
)
_sys.modules[__name__] = _implementation
