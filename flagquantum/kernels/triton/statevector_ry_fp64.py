"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_ry_fp64 import __all__ as __all__
from flagquantum.simulation.kernels.triton.statevector_ry_fp64 import (
    _ry as _ry,
)
from flagquantum.simulation.kernels.triton.statevector_ry_fp64 import (
    _validate as _validate,
)
from flagquantum.simulation.kernels.triton.statevector_ry_fp64 import (
    _vjp as _vjp,
)
from flagquantum.simulation.kernels.triton.statevector_ry_fp64 import (
    backward as backward,
)
from flagquantum.simulation.kernels.triton.statevector_ry_fp64 import (
    forward as forward,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_ry_fp64"
)
_sys.modules[__name__] = _implementation
