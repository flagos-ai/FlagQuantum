"""Historical imports for reed muller definitions; Core owns these aliases."""

import itertools as itertools
from numbers import Integral as Integral

from flagquantum.qec.codes.css import CssCode as CssCode
from flagquantum.qec.codes.qldpc import _logical_basis as _logical_basis
from flagquantum.qec.codes.qldpc import _paired_basis as _paired_basis
from flagquantum.qec.codes.qldpc import _rows as _rows
from flagquantum.qec.gf2 import nullspace as nullspace

from .codes.reed_muller import _LARGEST_VARIABLES as _LARGEST_VARIABLES
from .codes.reed_muller import _SEARCH_BOUND as _SEARCH_BOUND
from .codes.reed_muller import _SMALLEST_VARIABLES as _SMALLEST_VARIABLES
from .codes.reed_muller import __all__ as __all__
from .codes.reed_muller import _monomial_rows as _monomial_rows
from .codes.reed_muller import _punctured as _punctured
from .codes.reed_muller import reed_muller_code as reed_muller_code
