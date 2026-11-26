"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.transforms import (
    BoundedLoopUnrollPass as BoundedLoopUnrollPass,
)
from flagquantum.compiler.frontends._structured.transforms import (
    ConstantFoldPass as ConstantFoldPass,
)
from flagquantum.compiler.frontends._structured.transforms import (
    DeadConstantEliminationPass as DeadConstantEliminationPass,
)
from flagquantum.compiler.frontends._structured.transforms import (
    PassOutcome as PassOutcome,
)
from flagquantum.compiler.frontends._structured.transforms import (
    StructuredControlFlowSimplificationPass as StructuredControlFlowSimplificationPass,
)
from flagquantum.compiler.frontends._structured.transforms import __all__ as __all__

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.transforms"
)
_sys.modules[__name__] = _implementation
