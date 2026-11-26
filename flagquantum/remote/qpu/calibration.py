"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.calibration import __all__ as __all__
from flagquantum.compute.remote.qpu.calibration import (
    _captured_at as _captured_at,
)
from flagquantum.compute.remote.qpu.calibration import (
    quafu_noise_model_from_chip_info as quafu_noise_model_from_chip_info,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.calibration")
_sys.modules[__name__] = _implementation
