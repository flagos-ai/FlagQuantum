"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.quantum_ml.encoding import (
    _DEFAULT_ROTATION as _DEFAULT_ROTATION,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    _ROTATION_GATES as _ROTATION_GATES,
)
from flagquantum.algorithms.quantum_ml.encoding import __all__ as __all__
from flagquantum.algorithms.quantum_ml.encoding import (
    _amplitude_vector as _amplitude_vector,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    _as_tensor as _as_tensor,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    _resolve_qubits as _resolve_qubits,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    _rotation_angles as _rotation_angles,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    _rotation_gate as _rotation_gate,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    amplitude_encode as amplitude_encode,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    angular_encode as angular_encode,
)
from flagquantum.algorithms.quantum_ml.encoding import (
    append_angular_encode as append_angular_encode,
)

_implementation = _import_module("flagquantum.algorithms.quantum_ml.encoding")
_sys.modules[__name__] = _implementation
