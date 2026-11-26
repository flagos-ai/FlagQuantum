"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.search.grover import (
    _GROVER_QUBIT_LIMIT as _GROVER_QUBIT_LIMIT,
)
from flagquantum.algorithms.search.grover import (
    GroverResult as GroverResult,
)
from flagquantum.algorithms.search.grover import __all__ as __all__
from flagquantum.algorithms.search.grover import (
    _append_diffusion as _append_diffusion,
)
from flagquantum.algorithms.search.grover import (
    _append_multi_controlled_z as _append_multi_controlled_z,
)
from flagquantum.algorithms.search.grover import (
    _validate_register_width as _validate_register_width,
)
from flagquantum.algorithms.search.grover import (
    _validate_search_width as _validate_search_width,
)
from flagquantum.algorithms.search.grover import (
    grover_circuit as grover_circuit,
)
from flagquantum.algorithms.search.grover import (
    optimal_iterations as optimal_iterations,
)
from flagquantum.algorithms.search.grover import (
    run_grover as run_grover,
)

_implementation = _import_module("flagquantum.algorithms.search.grover")
_sys.modules[__name__] = _implementation
