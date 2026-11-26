"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.flagtree import (
    _TLE_BACKEND_ALIASES as _TLE_BACKEND_ALIASES,
)
from flagquantum.simulation.kernels.flagtree import __all__ as __all__
from flagquantum.simulation.kernels.flagtree import (
    apply_complex64_local_1q_tle as apply_complex64_local_1q_tle,
)
from flagquantum.simulation.kernels.flagtree import (
    apply_complex64_transpose_1q_tle_inplace as apply_complex64_transpose_1q_tle_inplace,
)
from flagquantum.simulation.kernels.flagtree import (
    fused_complex64_sharded_1q_vjp_adjoint_tle as fused_complex64_sharded_1q_vjp_adjoint_tle,
)
from flagquantum.simulation.kernels.flagtree import (
    pack_complex64_control_one_tle as pack_complex64_control_one_tle,
)
from flagquantum.simulation.kernels.flagtree import (
    require_flagtree_tle_primitive as require_flagtree_tle_primitive,
)
from flagquantum.simulation.kernels.flagtree import (
    unpack_complex64_control_one_tle as unpack_complex64_control_one_tle,
)

_implementation = _import_module("flagquantum.simulation.kernels.flagtree")
_sys.modules[__name__] = _implementation
