"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.core.noise import (
    CorrelatedReadoutError as CorrelatedReadoutError,
)
from flagquantum.core.noise import (
    NoiseModel as NoiseModel,
)
from flagquantum.core.noise import (
    NoiseRule as NoiseRule,
)
from flagquantum.core.noise import (
    ReadoutError as ReadoutError,
)
from flagquantum.core.noise import (
    ReadoutRule as ReadoutRule,
)
from flagquantum.core.noise import __all__ as __all__
from flagquantum.core.noise import (
    _decode_readout_error as _decode_readout_error,
)

_implementation = _import_module("flagquantum.core.noise")
_sys.modules[__name__] = _implementation
