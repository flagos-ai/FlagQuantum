"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.quantum_ml.kmedians import (
    _CENTROID_WIRE_LIMIT as _CENTROID_WIRE_LIMIT,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _DEFAULT_SHOTS as _DEFAULT_SHOTS,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _MAX_CENTROIDS as _MAX_CENTROIDS,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    KMediansResult as KMediansResult,
)
from flagquantum.algorithms.quantum_ml.kmedians import __all__ as __all__
from flagquantum.algorithms.quantum_ml.kmedians import (
    _centroid_index_width as _centroid_index_width,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _coordinates as _coordinates,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _median_positions as _median_positions,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _nearest_centroid as _nearest_centroid,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _operands as _operands,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    _strictly_better as _strictly_better,
)
from flagquantum.algorithms.quantum_ml.kmedians import (
    kmedians as kmedians,
)

_implementation = _import_module("flagquantum.algorithms.quantum_ml.kmedians")
_sys.modules[__name__] = _implementation
