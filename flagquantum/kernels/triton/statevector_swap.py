"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_swap import __all__ as __all__
from flagquantum.simulation.kernels.triton.statevector_swap import (
    _complex64_local_swap_sequence_kernel as _complex64_local_swap_sequence_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_swap import (
    _swap_source_bit_positions as _swap_source_bit_positions,
)
from flagquantum.simulation.kernels.triton.statevector_swap import (
    apply_complex64_local_swap_sequence as apply_complex64_local_swap_sequence,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_swap"
)
_sys.modules[__name__] = _implementation
