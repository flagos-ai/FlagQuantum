"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.analysis import (
    FOLDABLE_OPERATIONS as FOLDABLE_OPERATIONS,
)
from flagquantum.compiler.frontends._structured.analysis import (
    HybridProgramAnalysis as HybridProgramAnalysis,
)
from flagquantum.compiler.frontends._structured.analysis import __all__ as __all__
from flagquantum.compiler.frontends._structured.analysis import (
    analyze_program as analyze_program,
)
from flagquantum.compiler.frontends._structured.analysis import (
    constant_loop_iterations as constant_loop_iterations,
)
from flagquantum.compiler.frontends._structured.analysis import (
    direct_constant_loop_counts as direct_constant_loop_counts,
)
from flagquantum.compiler.frontends._structured.analysis import (
    fold_value as fold_value,
)
from flagquantum.compiler.frontends._structured.analysis import (
    operations as operations,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured.analysis")
_sys.modules[__name__] = _implementation
