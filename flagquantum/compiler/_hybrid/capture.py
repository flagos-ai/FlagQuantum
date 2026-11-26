"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

_implementation = _import_module("flagquantum.compiler.frontends._structured.capture")
_sys.modules[__name__] = _implementation
