"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _BLOCK_SIZE as _BLOCK_SIZE,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _MAX_SITE_ELEMENTS as _MAX_SITE_ELEMENTS,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _mps_qubit_probabilities_kernel as _mps_qubit_probabilities_kernel,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _reference as _reference,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _supported_shape as _supported_shape,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    _validate as _validate,
)
from flagquantum.simulation.kernels.triton.mps_wire_probabilities import (
    fused_mps_qubit_probabilities as fused_mps_qubit_probabilities,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.mps_wire_probabilities"
)
_sys.modules[__name__] = _implementation
