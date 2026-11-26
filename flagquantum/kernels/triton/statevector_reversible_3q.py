"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_reversible_3q import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.statevector_reversible_3q import (
    _complex64_local_reversible_3q_kernel as _complex64_local_reversible_3q_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_reversible_3q import (
    apply_complex64_local_reversible_3q as apply_complex64_local_reversible_3q,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_reversible_3q"
)
_sys.modules[__name__] = _implementation
