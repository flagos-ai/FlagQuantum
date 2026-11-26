"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization.qaoa import (
    QAOA_ASSUMPTIONS as QAOA_ASSUMPTIONS,
)
from flagquantum.algorithms.optimization.qaoa import (
    QAOA_LIMITATIONS as QAOA_LIMITATIONS,
)
from flagquantum.algorithms.optimization.qaoa import (
    QAOAResult as QAOAResult,
)
from flagquantum.algorithms.optimization.qaoa import __all__ as __all__
from flagquantum.algorithms.optimization.qaoa import (
    _checked_cut_edges as _checked_cut_edges,
)
from flagquantum.algorithms.optimization.qaoa import (
    _checked_endpoint as _checked_endpoint,
)
from flagquantum.algorithms.optimization.qaoa import (
    maxcut_hamiltonian as maxcut_hamiltonian,
)
from flagquantum.algorithms.optimization.qaoa import (
    maxcut_objective as maxcut_objective,
)
from flagquantum.algorithms.optimization.qaoa import (
    run_qaoa as run_qaoa,
)

_implementation = _import_module("flagquantum.algorithms.optimization.qaoa")
_sys.modules[__name__] = _implementation
