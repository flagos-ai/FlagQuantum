"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.core.operators.superoperator import (
    _BASIS_TOLERANCE_IN_EPSILON as _BASIS_TOLERANCE_IN_EPSILON,
)
from flagquantum.core.operators.superoperator import (
    _CHOI_TOLERANCE_IN_EPSILON as _CHOI_TOLERANCE_IN_EPSILON,
)
from flagquantum.core.operators.superoperator import (
    DEFAULT_DENSE_MATRIX_BYTES as DEFAULT_DENSE_MATRIX_BYTES,
)
from flagquantum.core.operators.superoperator import (
    Factor as Factor,
)
from flagquantum.core.operators.superoperator import (
    OperatorFactor as OperatorFactor,
)
from flagquantum.core.operators.superoperator import (
    SuperOperator as SuperOperator,
)
from flagquantum.core.operators.superoperator import __all__ as __all__
from flagquantum.core.operators.superoperator import (
    _basis_entry as _basis_entry,
)
from flagquantum.core.operators.superoperator import (
    _choi_eigenvalues as _choi_eigenvalues,
)
from flagquantum.core.operators.superoperator import (
    _coefficient as _coefficient,
)
from flagquantum.core.operators.superoperator import (
    _dense_matrix as _dense_matrix,
)
from flagquantum.core.operators.superoperator import (
    _factor_adjoint as _factor_adjoint,
)
from flagquantum.core.operators.superoperator import (
    _factor_dimension as _factor_dimension,
)
from flagquantum.core.operators.superoperator import (
    _factor_dtype as _factor_dtype,
)
from flagquantum.core.operators.superoperator import (
    _same_factor as _same_factor,
)
from flagquantum.core.operators.superoperator import (
    _Term as _Term,
)

_implementation = _import_module("flagquantum.core.operators.superoperator")
_sys.modules[__name__] = _implementation
