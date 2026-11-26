"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.quantum_ml.models import (
    HybridQuantumClassifier as HybridQuantumClassifier,
)
from flagquantum.algorithms.quantum_ml.models import (
    VariationalEnergyModel as VariationalEnergyModel,
)
from flagquantum.algorithms.quantum_ml.models import __all__ as __all__
from flagquantum.algorithms.quantum_ml.models import (
    _classifier_circuit as _classifier_circuit,
)
from flagquantum.algorithms.quantum_ml.models import (
    _deployment_quantum_parameters as _deployment_quantum_parameters,
)
from flagquantum.algorithms.quantum_ml.models import (
    _energy_circuit as _energy_circuit,
)

_implementation = _import_module("flagquantum.algorithms.quantum_ml.models")
_sys.modules[__name__] = _implementation
