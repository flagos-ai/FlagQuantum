"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization.solvers import (
    Ansatz as Ansatz,
)
from flagquantum.algorithms.optimization.solvers import (
    VQESolver as VQESolver,
)
from flagquantum.algorithms.optimization.solvers import __all__ as __all__
from flagquantum.algorithms.optimization.solvers import (
    vqe_objective as vqe_objective,
)

_implementation = _import_module("flagquantum.algorithms.optimization.solvers")
_sys.modules[__name__] = _implementation
