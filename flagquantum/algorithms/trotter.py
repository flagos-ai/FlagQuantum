"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.many_body.trotter import (
    _BASIS_CHANGE as _BASIS_CHANGE,
)
from flagquantum.algorithms.many_body.trotter import (
    TROTTER_ORDERS as TROTTER_ORDERS,
)
from flagquantum.algorithms.many_body.trotter import __all__ as __all__
from flagquantum.algorithms.many_body.trotter import (
    _append_exponential as _append_exponential,
)
from flagquantum.algorithms.many_body.trotter import (
    _append_term as _append_term,
)
from flagquantum.algorithms.many_body.trotter import (
    _prepare_terms as _prepare_terms,
)
from flagquantum.algorithms.many_body.trotter import (
    _real_coefficient as _real_coefficient,
)
from flagquantum.algorithms.many_body.trotter import (
    _require_order as _require_order,
)
from flagquantum.algorithms.many_body.trotter import (
    _require_steps as _require_steps,
)
from flagquantum.algorithms.many_body.trotter import (
    _supported_operators as _supported_operators,
)
from flagquantum.algorithms.many_body.trotter import (
    pauli_exponential_circuit as pauli_exponential_circuit,
)
from flagquantum.algorithms.many_body.trotter import (
    trotter_circuit as trotter_circuit,
)

_implementation = _import_module("flagquantum.algorithms.many_body.trotter")
_sys.modules[__name__] = _implementation
