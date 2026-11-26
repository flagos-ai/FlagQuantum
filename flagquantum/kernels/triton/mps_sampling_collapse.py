"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    _MAX_BOND as _MAX_BOND,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    __all__ as __all__,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    _mps_sampling_collapse_kernel as _mps_sampling_collapse_kernel,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    _reference as _reference,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    _supported_shape as _supported_shape,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    _validate as _validate,
)
from flagquantum.simulation.kernels.triton.mps_sampling_collapse import (
    fused_mps_sampling_collapse as fused_mps_sampling_collapse,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.mps_sampling_collapse"
)
_sys.modules[__name__] = _implementation
