"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.dynamic.program import (
    _EFFECT as _EFFECT,
)
from flagquantum.compiler.frontends._structured.dynamic.program import (
    LoweredDynamicProgram as LoweredDynamicProgram,
)
from flagquantum.compiler.frontends._structured.dynamic.program import (
    __all__ as __all__,
)
from flagquantum.compiler.frontends._structured.dynamic.program import (
    _DynamicLowerer as _DynamicLowerer,
)
from flagquantum.compiler.frontends._structured.dynamic.program import (
    lower_dynamic_program as lower_dynamic_program,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.dynamic.program"
)
_sys.modules[__name__] = _implementation
