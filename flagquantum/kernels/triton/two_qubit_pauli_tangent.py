"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.two_qubit_pauli_tangent import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.two_qubit_pauli_tangent import (
    _reference as _reference,
)
from flagquantum.simulation.kernels.triton.two_qubit_pauli_tangent import (
    _rxx_ryy_rzz_tangent_kernel as _rxx_ryy_rzz_tangent_kernel,
)
from flagquantum.simulation.kernels.triton.two_qubit_pauli_tangent import (
    _TensorJacobian as _TensorJacobian,
)
from flagquantum.simulation.kernels.triton.two_qubit_pauli_tangent import (
    repeated_rxx_ryy_rzz_tangents as repeated_rxx_ryy_rzz_tangents,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.two_qubit_pauli_tangent"
)
_sys.modules[__name__] = _implementation
