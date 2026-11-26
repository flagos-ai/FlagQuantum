"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.mps_one_site import __all__ as __all__
from flagquantum.simulation.kernels.triton.mps_one_site import (
    _FusedMPSOneSite as _FusedMPSOneSite,
)
from flagquantum.simulation.kernels.triton.mps_one_site import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.mps_one_site import (
    _one_site_forward_kernel as _one_site_forward_kernel,
)
from flagquantum.simulation.kernels.triton.mps_one_site import (
    _OneSiteContext as _OneSiteContext,
)
from flagquantum.simulation.kernels.triton.mps_one_site import (
    _reference as _reference,
)
from flagquantum.simulation.kernels.triton.mps_one_site import (
    fused_mps_one_site as fused_mps_one_site,
)

_implementation = _import_module("flagquantum.simulation.kernels.triton.mps_one_site")
_sys.modules[__name__] = _implementation
