"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.cpu_target_capabilities import (
    _PRECISION_FACTS as _PRECISION_FACTS,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    CPU_PLATFORM_ADAPTER_VERSION as CPU_PLATFORM_ADAPTER_VERSION,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    CPU_PLATFORM_PROVIDER as CPU_PLATFORM_PROVIDER,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    CPU_TARGET_CLASS as CPU_TARGET_CLASS,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    CPUCapabilityObservation as CPUCapabilityObservation,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    CPUCapabilityProbe as CPUCapabilityProbe,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    CPUPrecisionObservation as CPUPrecisionObservation,
)
from flagquantum.compute.local.cpu_target_capabilities import __all__ as __all__
from flagquantum.compute.local.cpu_target_capabilities import (
    _blocker as _blocker,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    _isoformat as _isoformat,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    _missing_fact as _missing_fact,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    _ObservedCPUProbe as _ObservedCPUProbe,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    cpu_platform_to_target_capability_snapshot as cpu_platform_to_target_capability_snapshot,
)
from flagquantum.compute.local.cpu_target_capabilities import (
    probe_local_cpu_target_capabilities as probe_local_cpu_target_capabilities,
)

_implementation = _import_module("flagquantum.compute.local.cpu_target_capabilities")
_sys.modules[__name__] = _implementation
