"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.results.mps import (
    mps_marginal as mps_marginal,
)
from flagquantum.runtime.results.mps import (
    native_mps as native_mps,
)

_implementation = _import_module("flagquantum.runtime.results.mps")
_sys.modules[__name__] = _implementation
