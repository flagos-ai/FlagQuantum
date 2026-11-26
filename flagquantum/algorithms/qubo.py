"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization.qubo import (
    _IDENTITY_WIRE as _IDENTITY_WIRE,
)
from flagquantum.algorithms.optimization.qubo import (
    QuboProblem as QuboProblem,
)
from flagquantum.algorithms.optimization.qubo import __all__ as __all__
from flagquantum.algorithms.optimization.qubo import (
    _constant_wire as _constant_wire,
)
from flagquantum.algorithms.optimization.qubo import (
    _is_index as _is_index,
)
from flagquantum.algorithms.optimization.qubo import (
    ising_to_qubo as ising_to_qubo,
)
from flagquantum.algorithms.optimization.qubo import (
    max_cut_qubo as max_cut_qubo,
)
from flagquantum.algorithms.optimization.qubo import (
    qubo_energy as qubo_energy,
)
from flagquantum.algorithms.optimization.qubo import (
    qubo_to_ising as qubo_to_ising,
)

_implementation = _import_module("flagquantum.algorithms.optimization.qubo")
_sys.modules[__name__] = _implementation
