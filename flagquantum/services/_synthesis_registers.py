"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.synth._program import (
    _BUDGET as _BUDGET,
)
from flagquantum.synth._program import (
    _CONSTRAINTS as _CONSTRAINTS,
)
from flagquantum.synth._program import (
    _TARGET as _TARGET,
)
from flagquantum.synth._program import (
    synthesize_registers as synthesize_registers,
)

_implementation = _import_module("flagquantum.synth._program")
_sys.modules[__name__] = _implementation
