"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu._quafu_physical import (
    _CREG_DECLARATION as _CREG_DECLARATION,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    _CX_STATEMENT as _CX_STATEMENT,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    _GATE_STATEMENT as _GATE_STATEMENT,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    _MEASUREMENT as _MEASUREMENT,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    _QREG_DECLARATION as _QREG_DECLARATION,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    _QUBIT_REFERENCE as _QUBIT_REFERENCE,
)
from flagquantum.compute.remote.qpu._quafu_physical import __all__ as __all__
from flagquantum.compute.remote.qpu._quafu_physical import (
    backend_snapshot_sha256 as backend_snapshot_sha256,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    prepare_physical_qasm as prepare_physical_qasm,
)
from flagquantum.compute.remote.qpu._quafu_physical import (
    validate_physical_result as validate_physical_result,
)

_implementation = _import_module("flagquantum.compute.remote.qpu._quafu_physical")
_sys.modules[__name__] = _implementation
