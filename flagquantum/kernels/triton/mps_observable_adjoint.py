"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _MAX_BOND as _MAX_BOND,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _MAX_CONTRACTION_WORK as _MAX_CONTRACTION_WORK,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _mps_hermitian_observable_adjoint_kernel as _mps_hermitian_observable_adjoint_kernel,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _reference as _reference,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _supported_shape as _supported_shape,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    _validate as _validate,
)
from flagquantum.simulation.kernels.triton.mps_observable_adjoint import (
    fused_mps_hermitian_observable_adjoint as fused_mps_hermitian_observable_adjoint,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.mps_observable_adjoint"
)
_sys.modules[__name__] = _implementation
