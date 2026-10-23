"""Historical imports for floquet definitions; Core owns these aliases."""

import itertools as itertools
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from numbers import Integral as Integral

from flagquantum.qec.codes._css_helpers import (
    _DISTANCE_SEARCH_WEIGHT as _DISTANCE_SEARCH_WEIGHT,
)
from flagquantum.qec.gf2 import in_span as in_span
from flagquantum.qec.gf2 import nullspace as nullspace
from flagquantum.qec.gf2 import rank as rank
from flagquantum.qec.gf2 import reduce_rows as reduce_rows
from flagquantum.qec.logical import _symplectic_mask as _symplectic_mask
from flagquantum.qec.pauli import Pauli as Pauli

from .codes.floquet import _DEFAULT_MAX_ROUNDS as _DEFAULT_MAX_ROUNDS
from .codes.floquet import FloquetCode as FloquetCode
from .codes.floquet import MeasurementPhase as MeasurementPhase
from .codes.floquet import __all__ as __all__
from .codes.floquet import _advance as _advance
from .codes.floquet import _centralizer as _centralizer
from .codes.floquet import _combination as _combination
from .codes.floquet import _commutes as _commutes
from .codes.floquet import _least_logical_weight as _least_logical_weight
from .codes.floquet import _phase_rows as _phase_rows
from .codes.floquet import _quotient_basis as _quotient_basis
from .codes.floquet import _ring_edges as _ring_edges
from .codes.floquet import _same_span as _same_span
from .codes.floquet import _settle as _settle
from .codes.floquet import _support as _support
from .codes.floquet import _surviving as _surviving
from .codes.floquet import _to_pauli as _to_pauli
from .codes.floquet import ring_floquet_code as ring_floquet_code
