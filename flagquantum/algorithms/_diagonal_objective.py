"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._diagonal_objective import (
    DiagonalMetrics as DiagonalMetrics,
)
from flagquantum.algorithms.optimization._diagonal_objective import (
    DiagonalQaoa as DiagonalQaoa,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._diagonal_objective"
)
_sys.modules[__name__] = _implementation
