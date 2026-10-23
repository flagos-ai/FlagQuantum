"""Historical imports for bicycle definitions; Core owns these aliases."""

from collections.abc import Sequence as Sequence
from numbers import Integral as Integral

from flagquantum.qec.codes._css_helpers import (
    _DISTANCE_SEARCH_WEIGHT as _DISTANCE_SEARCH_WEIGHT,
)
from flagquantum.qec.codes.css import CssCode as CssCode
from flagquantum.qec.codes.qldpc import _logical_basis as _logical_basis
from flagquantum.qec.codes.qldpc import _paired_basis as _paired_basis
from flagquantum.qec.codes.qldpc import _rows as _rows

from .codes.bicycle import __all__ as __all__
from .codes.bicycle import _monomial_exponents as _monomial_exponents
from .codes.bicycle import _polynomial_masks as _polynomial_masks
from .codes.bicycle import _transpose as _transpose
from .codes.bicycle import bivariate_bicycle_code as bivariate_bicycle_code
