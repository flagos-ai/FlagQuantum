"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu._quafu_calibration import __all__ as __all__
from flagquantum.compute.remote.qpu._quafu_calibration import (
    _fidelity as _fidelity,
)
from flagquantum.compute.remote.qpu._quafu_calibration import (
    _is_sequence as _is_sequence,
)
from flagquantum.compute.remote.qpu._quafu_calibration import (
    _qubit_index as _qubit_index,
)
from flagquantum.compute.remote.qpu._quafu_calibration import (
    qsteed_chip_info as qsteed_chip_info,
)

_implementation = _import_module("flagquantum.compute.remote.qpu._quafu_calibration")
_sys.modules[__name__] = _implementation
