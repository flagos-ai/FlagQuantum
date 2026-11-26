"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.execution import __all__ as __all__
from flagquantum.compute.remote.qpu.execution import (
    _azure_target_width as _azure_target_width,
)
from flagquantum.compute.remote.qpu.execution import (
    _execute_counts as _execute_counts,
)
from flagquantum.compute.remote.qpu.execution import (
    _execute_expectation as _execute_expectation,
)
from flagquantum.compute.remote.qpu.execution import (
    _pauli_output_observable as _pauli_output_observable,
)
from flagquantum.compute.remote.qpu.execution import (
    counts_result as counts_result,
)
from flagquantum.compute.remote.qpu.execution import (
    execute_azure as execute_azure,
)
from flagquantum.compute.remote.qpu.execution import (
    execute_quafu as execute_quafu,
)
from flagquantum.compute.remote.qpu.execution import (
    validate_azure_output as validate_azure_output,
)
from flagquantum.compute.remote.qpu.execution import (
    validate_quafu_output as validate_quafu_output,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.execution")
_sys.modules[__name__] = _implementation
