"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.lowering import (
    CircuitStructureCache as CircuitStructureCache,
)
from flagquantum.compiler.frontends._structured.lowering import (
    LoweredHybridProgram as LoweredHybridProgram,
)
from flagquantum.compiler.frontends._structured.lowering import (
    _template as _template,
)
from flagquantum.compiler.frontends._structured.lowering import (
    lower_trace as lower_trace,
)
from flagquantum.compiler.frontends._structured.lowering import (
    specialize_and_lower as specialize_and_lower,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured.lowering")
_sys.modules[__name__] = _implementation
