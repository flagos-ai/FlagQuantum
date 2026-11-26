"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.catalog.schema import (
    KERNEL_DEVICES as KERNEL_DEVICES,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelDevice as KernelDevice,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelDirection as KernelDirection,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelDomain as KernelDomain,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelEvidence as KernelEvidence,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelImplementation as KernelImplementation,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelMaturity as KernelMaturity,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelProvider as KernelProvider,
)
from flagquantum.simulation.kernels.catalog.schema import (
    KernelSemantic as KernelSemantic,
)
from flagquantum.simulation.kernels.catalog.schema import __all__ as __all__

_implementation = _import_module("flagquantum.simulation.kernels.catalog.schema")
_sys.modules[__name__] = _implementation
