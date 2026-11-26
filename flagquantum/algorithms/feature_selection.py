"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.quantum_ml.feature_selection import (
    FeatureSelectionProblem as FeatureSelectionProblem,
)
from flagquantum.algorithms.quantum_ml.feature_selection import __all__ as __all__
from flagquantum.algorithms.quantum_ml.feature_selection import (
    _validated_penalty as _validated_penalty,
)
from flagquantum.algorithms.quantum_ml.feature_selection import (
    _validated_redundancy as _validated_redundancy,
)
from flagquantum.algorithms.quantum_ml.feature_selection import (
    _validated_relevance as _validated_relevance,
)
from flagquantum.algorithms.quantum_ml.feature_selection import (
    _validated_target as _validated_target,
)
from flagquantum.algorithms.quantum_ml.feature_selection import (
    feature_selection_qubo as feature_selection_qubo,
)

_implementation = _import_module("flagquantum.algorithms.quantum_ml.feature_selection")
_sys.modules[__name__] = _implementation
