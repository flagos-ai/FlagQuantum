"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimizers.cobyla import (
    _MAX_PIVOTS as _MAX_PIVOTS,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _PIVOT_TOLERANCE as _PIVOT_TOLERANCE,
)
from flagquantum.algorithms.optimizers.cobyla import (
    COBYLA_ASSUMPTIONS as COBYLA_ASSUMPTIONS,
)
from flagquantum.algorithms.optimizers.cobyla import (
    COBYLA_LIMITATIONS as COBYLA_LIMITATIONS,
)
from flagquantum.algorithms.optimizers.cobyla import (
    CobylaOptimizer as CobylaOptimizer,
)
from flagquantum.algorithms.optimizers.cobyla import (
    CobylaResult as CobylaResult,
)
from flagquantum.algorithms.optimizers.cobyla import (
    Constraint as Constraint,
)
from flagquantum.algorithms.optimizers.cobyla import (
    Objective as Objective,
)
from flagquantum.algorithms.optimizers.cobyla import __all__ as __all__
from flagquantum.algorithms.optimizers.cobyla import (
    _AffineModel as _AffineModel,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _finite_positive as _finite_positive,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _initial_tableau as _initial_tableau,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _largest_violation as _largest_violation,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _linearized_residual as _linearized_residual,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _residual as _residual,
)
from flagquantum.algorithms.optimizers.cobyla import (
    _trust_region_step as _trust_region_step,
)

_implementation = _import_module("flagquantum.algorithms.optimizers.cobyla")
_sys.modules[__name__] = _implementation
