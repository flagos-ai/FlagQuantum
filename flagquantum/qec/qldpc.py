"""Historical imports for qldpc definitions; Core owns these aliases."""

from collections.abc import Sequence as Sequence

from flagquantum.qec.codes._css_helpers import (
    _DISTANCE_SEARCH_WEIGHT as _DISTANCE_SEARCH_WEIGHT,
)
from flagquantum.qec.codes._css_helpers import BinaryMatrix as BinaryMatrix
from flagquantum.qec.codes._css_helpers import _block_width as _block_width
from flagquantum.qec.codes._css_helpers import _matrix_rows as _matrix_rows
from flagquantum.qec.codes.css import CssCode as CssCode
from flagquantum.qec.gf2 import in_span as in_span
from flagquantum.qec.gf2 import nullspace as nullspace
from flagquantum.qec.gf2 import reduce_rows as reduce_rows

from .codes.qldpc import __all__ as __all__
from .codes.qldpc import _logical_basis as _logical_basis
from .codes.qldpc import _masks as _masks
from .codes.qldpc import _paired_basis as _paired_basis
from .codes.qldpc import _rows as _rows
from .codes.qldpc import qldpc_code as qldpc_code
