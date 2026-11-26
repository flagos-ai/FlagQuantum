"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.synthesis.programs.preparation import (
    prepare_inputs as prepare_inputs,
)

_implementation = _import_module("flagquantum.compiler.synthesis.programs.preparation")
_sys.modules[__name__] = _implementation
