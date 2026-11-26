"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.unitary_effects import (
    PURE_EFFECTS as PURE_EFFECTS,
)
from flagquantum.compiler.frontends._structured.unitary_effects import (
    _inverse_specs as _inverse_specs,
)
from flagquantum.compiler.frontends._structured.unitary_effects import (
    inverse_effects as inverse_effects,
)
from flagquantum.compiler.frontends._structured.unitary_effects import (
    unitary_regions as unitary_regions,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.unitary_effects"
)
_sys.modules[__name__] = _implementation
