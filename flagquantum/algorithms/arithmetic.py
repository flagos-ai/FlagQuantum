"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.primitives.arithmetic import (
    _MINIMUM_BITS as _MINIMUM_BITS,
)
from flagquantum.algorithms.primitives.arithmetic import (
    AdderQubits as AdderQubits,
)
from flagquantum.algorithms.primitives.arithmetic import __all__ as __all__
from flagquantum.algorithms.primitives.arithmetic import (
    _require_bit_count as _require_bit_count,
)
from flagquantum.algorithms.primitives.arithmetic import (
    adder_circuit as adder_circuit,
)
from flagquantum.algorithms.primitives.arithmetic import (
    adder_qubits as adder_qubits,
)

_implementation = _import_module("flagquantum.algorithms.primitives.arithmetic")
_sys.modules[__name__] = _implementation
