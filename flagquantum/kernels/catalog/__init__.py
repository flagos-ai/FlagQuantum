"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.simulation.kernels.catalog import (
    EVIDENCE as EVIDENCE,
)
from flagquantum.simulation.kernels.catalog import (
    IMPLEMENTATIONS as IMPLEMENTATIONS,
)
from flagquantum.simulation.kernels.catalog import (
    KERNEL_DEVICES as KERNEL_DEVICES,
)
from flagquantum.simulation.kernels.catalog import (
    SEMANTICS as SEMANTICS,
)
from flagquantum.simulation.kernels.catalog import (
    KernelCandidate as KernelCandidate,
)
from flagquantum.simulation.kernels.catalog import (
    KernelDevice as KernelDevice,
)
from flagquantum.simulation.kernels.catalog import (
    KernelEvidence as KernelEvidence,
)
from flagquantum.simulation.kernels.catalog import (
    KernelImplementation as KernelImplementation,
)
from flagquantum.simulation.kernels.catalog import (
    KernelMatchResult as KernelMatchResult,
)
from flagquantum.simulation.kernels.catalog import (
    KernelMismatch as KernelMismatch,
)
from flagquantum.simulation.kernels.catalog import (
    KernelProvider as KernelProvider,
)
from flagquantum.simulation.kernels.catalog import (
    KernelRejection as KernelRejection,
)
from flagquantum.simulation.kernels.catalog import (
    KernelRequest as KernelRequest,
)
from flagquantum.simulation.kernels.catalog import (
    KernelSemantic as KernelSemantic,
)
from flagquantum.simulation.kernels.catalog import __all__ as __all__
from flagquantum.simulation.kernels.catalog import (
    match_kernel_implementations as match_kernel_implementations,
)
from flagquantum.simulation.kernels.catalog import (
    validate_catalog as validate_catalog,
)

_implementation = _import_module("flagquantum.simulation.kernels.catalog")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
