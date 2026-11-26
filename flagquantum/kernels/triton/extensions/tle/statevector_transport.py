"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.extensions.tle.statevector_transport import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_transport import (
    _complex64_control_one_pack_tle_kernel as _complex64_control_one_pack_tle_kernel,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_transport import (
    _complex64_control_one_unpack_tle_kernel as _complex64_control_one_unpack_tle_kernel,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_transport import (
    launch_complex64_control_one_pack_tle as launch_complex64_control_one_pack_tle,
)
from flagquantum.simulation.kernels.triton.extensions.tle.statevector_transport import (
    launch_complex64_control_one_unpack_tle as launch_complex64_control_one_unpack_tle,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.extensions.tle.statevector_transport"
)
_sys.modules[__name__] = _implementation
