"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_pauli_rotation import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.statevector_pauli_rotation import (
    _complex64_local_pauli_rotation_kernel as _complex64_local_pauli_rotation_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_pauli_rotation import (
    apply_complex64_local_pauli_rotation_2q as apply_complex64_local_pauli_rotation_2q,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_pauli_rotation"
)
_sys.modules[__name__] = _implementation
