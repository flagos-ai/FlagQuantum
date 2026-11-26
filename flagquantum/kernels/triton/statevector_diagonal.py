"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_diagonal import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.statevector_diagonal import (
    _complex64_local_diagonal_kernel as _complex64_local_diagonal_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_diagonal import (
    apply_complex64_local_diagonal as apply_complex64_local_diagonal,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_diagonal"
)
_sys.modules[__name__] = _implementation
