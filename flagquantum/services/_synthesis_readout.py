"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.synth._readout import (
    Distribution as Distribution,
)
from flagquantum.synth._readout import (
    _ReadDiagnostics as _ReadDiagnostics,
)
from flagquantum.synth._readout import (
    probabilities as probabilities,
)

_implementation = _import_module("flagquantum.synth._readout")
_sys.modules[__name__] = _implementation
