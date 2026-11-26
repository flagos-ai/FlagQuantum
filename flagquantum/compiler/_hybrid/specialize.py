"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.specialize import (
    _EFFECT as _EFFECT,
)
from flagquantum.compiler.frontends._structured.specialize import (
    _EXPECTATION as _EXPECTATION,
)
from flagquantum.compiler.frontends._structured.specialize import (
    SpecializedTrace as SpecializedTrace,
)
from flagquantum.compiler.frontends._structured.specialize import (
    TraceGate as TraceGate,
)
from flagquantum.compiler.frontends._structured.specialize import (
    _Specializer as _Specializer,
)
from flagquantum.compiler.frontends._structured.specialize import (
    specialize_program as specialize_program,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.specialize"
)
_sys.modules[__name__] = _implementation
