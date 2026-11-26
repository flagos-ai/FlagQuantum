"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.cuda_target_capabilities import (
    CUDA_PLATFORM_PROVIDER as CUDA_PLATFORM_PROVIDER,
)
from flagquantum.compute.local.cuda_target_capabilities import (
    CUDA_STATEVECTOR_WORKLOAD as CUDA_STATEVECTOR_WORKLOAD,
)
from flagquantum.compute.local.cuda_target_capabilities import __all__ as __all__
from flagquantum.compute.local.cuda_target_capabilities import (
    _timestamp as _timestamp,
)
from flagquantum.compute.local.cuda_target_capabilities import (
    cuda_statevector_capability_snapshot as cuda_statevector_capability_snapshot,
)

_implementation = _import_module("flagquantum.compute.local.cuda_target_capabilities")
_sys.modules[__name__] = _implementation
