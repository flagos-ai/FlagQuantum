"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.core.operators.fermion import (
    _ANNIHILATION_IMAGE as _ANNIHILATION_IMAGE,
)
from flagquantum.core.operators.fermion import (
    _CREATION_IMAGE as _CREATION_IMAGE,
)
from flagquantum.core.operators.fermion import (
    _IMAGINARY_TOLERANCE as _IMAGINARY_TOLERANCE,
)
from flagquantum.core.operators.fermion import (
    _PAULI_PRODUCT as _PAULI_PRODUCT,
)
from flagquantum.core.operators.fermion import (
    FermionOperator as FermionOperator,
)
from flagquantum.core.operators.fermion import (
    FermionTerm as FermionTerm,
)
from flagquantum.core.operators.fermion import (
    PauliFactors as PauliFactors,
)
from flagquantum.core.operators.fermion import __all__ as __all__
from flagquantum.core.operators.fermion import (
    _canonical_terms as _canonical_terms,
)
from flagquantum.core.operators.fermion import (
    _complex_scalar as _complex_scalar,
)
from flagquantum.core.operators.fermion import (
    _compose as _compose,
)
from flagquantum.core.operators.fermion import (
    _degree as _degree,
)
from flagquantum.core.operators.fermion import (
    _degrees as _degrees,
)
from flagquantum.core.operators.fermion import (
    _factor_sequence as _factor_sequence,
)
from flagquantum.core.operators.fermion import (
    _first_defect as _first_defect,
)
from flagquantum.core.operators.fermion import (
    _image_of as _image_of,
)
from flagquantum.core.operators.fermion import (
    _jordan_wigner_factor as _jordan_wigner_factor,
)
from flagquantum.core.operators.fermion import (
    _merge_pauli as _merge_pauli,
)
from flagquantum.core.operators.fermion import (
    _Monomial as _Monomial,
)
from flagquantum.core.operators.fermion import (
    _multiply_pauli as _multiply_pauli,
)
from flagquantum.core.operators.fermion import (
    _order_key as _order_key,
)
from flagquantum.core.operators.fermion import (
    _parity_factor as _parity_factor,
)
from flagquantum.core.operators.fermion import (
    _PauliSum as _PauliSum,
)
from flagquantum.core.operators.fermion import (
    annihilate as annihilate,
)
from flagquantum.core.operators.fermion import (
    create as create,
)
from flagquantum.core.operators.fermion import (
    jordan_wigner as jordan_wigner,
)
from flagquantum.core.operators.fermion import (
    number as number,
)
from flagquantum.core.operators.fermion import (
    parity_encoding as parity_encoding,
)

_implementation = _import_module("flagquantum.core.operators.fermion")
_sys.modules[__name__] = _implementation
