"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.complex_bmm import __all__ as __all__
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _BMMContext as _BMMContext,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _complex_bmm_kernel as _complex_bmm_kernel,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _complex_layout_bmm_kernel as _complex_layout_bmm_kernel,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _flattened_offset as _flattened_offset,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _FusedComplexBMM as _FusedComplexBMM,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _FusedComplexLayoutBMM as _FusedComplexLayoutBMM,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _launch_layout as _launch_layout,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    _LayoutBMMContext as _LayoutBMMContext,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    fused_complex_bmm as fused_complex_bmm,
)
from flagquantum.simulation.kernels.triton.complex_bmm import (
    fused_complex_layout_bmm as fused_complex_layout_bmm,
)

_implementation = _import_module("flagquantum.simulation.kernels.triton.complex_bmm")
_sys.modules[__name__] = _implementation
