"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.extensions.tle.statevector_adjoint import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_adjoint import (
    _complex64_sharded_1q_vjp_adjoint_tle_kernel as _complex64_sharded_1q_vjp_adjoint_tle_kernel,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_adjoint import (
    launch_complex64_sharded_1q_vjp_adjoint_tle as launch_complex64_sharded_1q_vjp_adjoint_tle,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.extensions.tle.statevector_adjoint"
)
_sys.modules[__name__] = _implementation
