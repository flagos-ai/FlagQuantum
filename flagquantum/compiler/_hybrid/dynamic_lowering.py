"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.dynamic_lowering import (
    LoweredDynamicProgram as LoweredDynamicProgram,
)
from flagquantum.compiler.frontends._structured.dynamic_lowering import (
    __all__ as __all__,
)
from flagquantum.compiler.frontends._structured.dynamic_lowering import (
    lower_dynamic_program as lower_dynamic_program,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.dynamic_lowering"
)
_sys.modules[__name__] = _implementation
