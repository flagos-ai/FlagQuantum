"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.core.operators.boson import (
    _DEFAULT_TRUNCATION_DIMENSION as _DEFAULT_TRUNCATION_DIMENSION,
)
from flagquantum.core.operators.boson import (
    DEFAULT_DENSE_MATRIX_BYTES as DEFAULT_DENSE_MATRIX_BYTES,
)
from flagquantum.core.operators.boson import (
    BosonOperator as BosonOperator,
)
from flagquantum.core.operators.boson import (
    BosonTerm as BosonTerm,
)
from flagquantum.core.operators.boson import __all__ as __all__
from flagquantum.core.operators.boson import (
    _canonical_terms as _canonical_terms,
)
from flagquantum.core.operators.boson import (
    _complex_scalar as _complex_scalar,
)
from flagquantum.core.operators.boson import (
    _compose as _compose,
)
from flagquantum.core.operators.boson import (
    _degree as _degree,
)
from flagquantum.core.operators.boson import (
    _degrees as _degrees,
)
from flagquantum.core.operators.boson import (
    _Factor as _Factor,
)
from flagquantum.core.operators.boson import (
    _factor_sequence as _factor_sequence,
)
from flagquantum.core.operators.boson import (
    _first_defect as _first_defect,
)
from flagquantum.core.operators.boson import (
    _ladder_matrices as _ladder_matrices,
)
from flagquantum.core.operators.boson import (
    _levels as _levels,
)
from flagquantum.core.operators.boson import (
    _Monomial as _Monomial,
)
from flagquantum.core.operators.boson import (
    _MonomialKey as _MonomialKey,
)
from flagquantum.core.operators.boson import (
    _term_matrix as _term_matrix,
)
from flagquantum.core.operators.boson import (
    _truncation as _truncation,
)
from flagquantum.core.operators.boson import (
    annihilate as annihilate,
)
from flagquantum.core.operators.boson import (
    create as create,
)
from flagquantum.core.operators.boson import (
    identity as identity,
)
from flagquantum.core.operators.boson import (
    momentum as momentum,
)
from flagquantum.core.operators.boson import (
    number as number,
)
from flagquantum.core.operators.boson import (
    position as position,
)

_implementation = _import_module("flagquantum.core.operators.boson")
_sys.modules[__name__] = _implementation
