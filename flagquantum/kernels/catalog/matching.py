"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.catalog.matching import (
    _MATURITY_RANK as _MATURITY_RANK,
)
from flagquantum.simulation.kernels.catalog.matching import (
    KernelCandidate as KernelCandidate,
)
from flagquantum.simulation.kernels.catalog.matching import (
    KernelMatchResult as KernelMatchResult,
)
from flagquantum.simulation.kernels.catalog.matching import (
    KernelMismatch as KernelMismatch,
)
from flagquantum.simulation.kernels.catalog.matching import (
    KernelMismatchCode as KernelMismatchCode,
)
from flagquantum.simulation.kernels.catalog.matching import (
    KernelRejection as KernelRejection,
)
from flagquantum.simulation.kernels.catalog.matching import (
    KernelRequest as KernelRequest,
)
from flagquantum.simulation.kernels.catalog.matching import __all__ as __all__
from flagquantum.simulation.kernels.catalog.matching import (
    _capability_mismatches as _capability_mismatches,
)
from flagquantum.simulation.kernels.catalog.matching import (
    _mismatch as _mismatch,
)
from flagquantum.simulation.kernels.catalog.matching import (
    match_kernel_implementations as match_kernel_implementations,
)

_implementation = _import_module("flagquantum.simulation.kernels.catalog.matching")
_sys.modules[__name__] = _implementation
