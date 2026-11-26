"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimizers.nelder_mead import (
    NELDER_MEAD_ASSUMPTIONS as NELDER_MEAD_ASSUMPTIONS,
)
from flagquantum.algorithms.optimizers.nelder_mead import (
    NELDER_MEAD_LIMITATIONS as NELDER_MEAD_LIMITATIONS,
)
from flagquantum.algorithms.optimizers.nelder_mead import (
    NelderMeadOptimizer as NelderMeadOptimizer,
)
from flagquantum.algorithms.optimizers.nelder_mead import (
    NelderMeadResult as NelderMeadResult,
)
from flagquantum.algorithms.optimizers.nelder_mead import (
    Objective as Objective,
)
from flagquantum.algorithms.optimizers.nelder_mead import __all__ as __all__
from flagquantum.algorithms.optimizers.nelder_mead import (
    _finite_positive as _finite_positive,
)

_implementation = _import_module("flagquantum.algorithms.optimizers.nelder_mead")
_sys.modules[__name__] = _implementation
