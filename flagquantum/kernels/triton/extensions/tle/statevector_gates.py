"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.extensions.tle.statevector_gates import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_gates import (
    _complex64_local_1q_tle_kernel as _complex64_local_1q_tle_kernel,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_gates import (
    _complex64_transpose_1q_tle_kernel as _complex64_transpose_1q_tle_kernel,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_gates import (
    launch_complex64_local_1q_tle as launch_complex64_local_1q_tle,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_gates import (
    launch_complex64_transpose_1q_tle as launch_complex64_transpose_1q_tle,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.extensions.tle.statevector_gates"
)
_sys.modules[__name__] = _implementation
