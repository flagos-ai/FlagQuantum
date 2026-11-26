"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.hva_forward_tangent import __all__ as __all__
from flagquantum.simulation.kernels.triton.hva_forward_tangent import (
    _insert_zero_bit as _insert_zero_bit,
)
from flagquantum.simulation.kernels.triton.hva_forward_tangent import (
    _rz_forward_tangent_kernel as _rz_forward_tangent_kernel,
)
from flagquantum.simulation.kernels.triton.hva_forward_tangent import (
    _two_qubit_forward_tangent_kernel as _two_qubit_forward_tangent_kernel,
)
from flagquantum.simulation.kernels.triton.hva_forward_tangent import (
    heisenberg_hva_forward_tangents as heisenberg_hva_forward_tangents,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.hva_forward_tangent"
)
_sys.modules[__name__] = _implementation
