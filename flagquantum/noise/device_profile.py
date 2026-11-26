"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.core.noise_profile import (
    DeviceNoiseProfile as DeviceNoiseProfile,
)
from flagquantum.core.noise_profile import (
    GateDuration as GateDuration,
)
from flagquantum.core.noise_profile import (
    QubitNoiseCalibration as QubitNoiseCalibration,
)
from flagquantum.core.noise_profile import __all__ as __all__

_implementation = _import_module("flagquantum.core.noise_profile")
_sys.modules[__name__] = _implementation
