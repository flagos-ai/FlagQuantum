"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.core.operators.pauli import (
    _PRODUCT as _PRODUCT,
)
from flagquantum.core.operators.pauli import (
    Factors as Factors,
)
from flagquantum.core.operators.pauli import (
    PauliOperator as PauliOperator,
)
from flagquantum.core.operators.pauli import (
    PauliTerm as PauliTerm,
)
from flagquantum.core.operators.pauli import (
    count as count,
)
from flagquantum.core.operators.pauli import (
    pauli as pauli,
)
from flagquantum.core.operators.pauli import (
    scalar as scalar,
)

_implementation = _import_module("flagquantum.core.operators.pauli")
_sys.modules[__name__] = _implementation
