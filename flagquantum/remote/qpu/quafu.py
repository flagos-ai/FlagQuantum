"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.quafu import (
    _MAX_SHOTS as _MAX_SHOTS,
)
from flagquantum.compute.remote.qpu.quafu import (
    _QREG_DECLARATION as _QREG_DECLARATION,
)
from flagquantum.compute.remote.qpu.quafu import (
    _QUBIT_REFERENCE as _QUBIT_REFERENCE,
)
from flagquantum.compute.remote.qpu.quafu import (
    _SDK_PROTOCOL as _SDK_PROTOCOL,
)
from flagquantum.compute.remote.qpu.quafu import (
    QuafuProvider as QuafuProvider,
)
from flagquantum.compute.remote.qpu.quafu import __all__ as __all__
from flagquantum.compute.remote.qpu.quafu import (
    _is_simulator_target as _is_simulator_target,
)
from flagquantum.compute.remote.qpu.quafu import (
    _quafu_backend_name as _quafu_backend_name,
)
from flagquantum.compute.remote.qpu.quafu import (
    _service_options as _service_options,
)
from flagquantum.compute.remote.qpu.quafu import (
    _submission_options as _submission_options,
)
from flagquantum.compute.remote.qpu.quafu import (
    _validate_quafu_shots as _validate_quafu_shots,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.quafu")
_sys.modules[__name__] = _implementation
