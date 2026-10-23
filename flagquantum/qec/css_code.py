"""Historical imports for css code definitions; Core owns these aliases."""

from dataclasses import dataclass as dataclass
from dataclasses import field as field
from numbers import Integral as Integral

import torch as torch

from flagquantum.qec.codes._css_helpers import (
    _DISTANCE_SEARCH_WEIGHT as _DISTANCE_SEARCH_WEIGHT,
)
from flagquantum.qec.codes._css_helpers import BinaryMatrix as BinaryMatrix
from flagquantum.qec.codes._css_helpers import _block_width as _block_width
from flagquantum.qec.codes._css_helpers import _mask as _mask
from flagquantum.qec.codes._css_helpers import _matrix_rows as _matrix_rows
from flagquantum.qec.codes._css_helpers import (
    _minimum_logical_weight as _minimum_logical_weight,
)
from flagquantum.qec.codes._css_helpers import (
    _refuse_non_logicals as _refuse_non_logicals,
)
from flagquantum.qec.codes._css_helpers import _supports as _supports
from flagquantum.qec.codes.checks import CodeCheck as CodeCheck
from flagquantum.qec.codes.checks import ancilla_band_counts as ancilla_band_counts
from flagquantum.qec.codes.matrices import CssCodeMatrices as CssCodeMatrices
from flagquantum.qec.gf2 import rank as rank
from flagquantum.qec.pauli import Pauli as Pauli

from .codes.css import CssCode as CssCode
from .codes.css import __all__ as __all__
from .codes.css import _in_span as _in_span
from .codes.css import _row_basis as _row_basis
from .codes.css import _row_masks as _row_masks
from .codes.css import _stated as _stated
from .codes.css import _support as _support
