"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.managed_simulator import (
    _QUAFU_MANAGED_SIMULATORS as _QUAFU_MANAGED_SIMULATORS,
)
from flagquantum.runtime.managed_simulator import (
    ManagedSimulatorExecution as ManagedSimulatorExecution,
)
from flagquantum.runtime.managed_simulator import (
    ManagedSimulatorReceipt as ManagedSimulatorReceipt,
)
from flagquantum.runtime.managed_simulator import __all__ as __all__
from flagquantum.runtime.managed_simulator import (
    managed_quafu_simulator_device as managed_quafu_simulator_device,
)
from flagquantum.runtime.managed_simulator import (
    run_managed_quafu_simulator as run_managed_quafu_simulator,
)

_implementation = _import_module("flagquantum.runtime.managed_simulator")
_sys.modules[__name__] = _implementation
