"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.linear_algebra.pca import (
    _DEFAULT_SHOTS as _DEFAULT_SHOTS,
)
from flagquantum.algorithms.linear_algebra.pca import (
    PcaResult as PcaResult,
)
from flagquantum.algorithms.linear_algebra.pca import __all__ as __all__
from flagquantum.algorithms.linear_algebra.pca import (
    _counter_distribution as _counter_distribution,
)
from flagquantum.algorithms.linear_algebra.pca import (
    _density_matrix as _density_matrix,
)
from flagquantum.algorithms.linear_algebra.pca import (
    _pca_circuit as _pca_circuit,
)
from flagquantum.algorithms.linear_algebra.pca import (
    _PhaseFromDensityMatrix as _PhaseFromDensityMatrix,
)
from flagquantum.algorithms.linear_algebra.pca import (
    _validate_counting_width as _validate_counting_width,
)
from flagquantum.algorithms.linear_algebra.pca import (
    principal_components as principal_components,
)

_implementation = _import_module("flagquantum.algorithms.linear_algebra.pca")
_sys.modules[__name__] = _implementation
