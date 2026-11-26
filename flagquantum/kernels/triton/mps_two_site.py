"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.mps_two_site import __all__ as __all__
from flagquantum.simulation.kernels.triton.mps_two_site import (
    _FusedMPSTwoSite as _FusedMPSTwoSite,
)
from flagquantum.simulation.kernels.triton.mps_two_site import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.mps_two_site import (
    _two_site_forward_kernel as _two_site_forward_kernel,
)
from flagquantum.simulation.kernels.triton.mps_two_site import (
    _two_site_range_kernel as _two_site_range_kernel,
)
from flagquantum.simulation.kernels.triton.mps_two_site import (
    _TwoSiteContext as _TwoSiteContext,
)
from flagquantum.simulation.kernels.triton.mps_two_site import (
    fused_mps_range_projection as fused_mps_range_projection,
)
from flagquantum.simulation.kernels.triton.mps_two_site import (
    fused_mps_two_site as fused_mps_two_site,
)

_implementation = _import_module("flagquantum.simulation.kernels.triton.mps_two_site")
_sys.modules[__name__] = _implementation
