"""Historical imports for subsystem definitions; Core owns these aliases."""

from dataclasses import dataclass as dataclass
from dataclasses import field as field
from numbers import Integral as Integral

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
from flagquantum.qec.gf2 import rank as rank
from flagquantum.qec.pauli import Pauli as Pauli

from .codes.subsystem import SubsystemCode as SubsystemCode
from .codes.subsystem import __all__ as __all__
from .codes.subsystem import tesseract_code as tesseract_code
from .codes.subsystem import tesseract_column_swap as tesseract_column_swap
from .codes.subsystem import tesseract_free_cnot_pairs as tesseract_free_cnot_pairs
