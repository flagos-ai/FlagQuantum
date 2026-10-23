"""Historical imports for :mod:`flagquantum.qec.dem.matrices`; Core-owned compatibility boundary."""

from __future__ import annotations

from numbers import Integral as Integral

import torch as torch

from flagquantum.qec.codes.checks import StabilizerCode as StabilizerCode
from flagquantum.qec.codes.matrices import CssCodeMatrices as CssCodeMatrices
from flagquantum.qec.codes.matrices import _binary_matrix as _binary_matrix
from flagquantum.qec.codes.matrices import css_code_matrices as css_code_matrices
from flagquantum.qec.dem.matrices import Entry as Entry
from flagquantum.qec.dem.matrices import __all__ as __all__
from flagquantum.qec.dem.matrices import _code_matrix_entries as _code_matrix_entries
from flagquantum.qec.dem.matrices import _column_supports as _column_supports
from flagquantum.qec.noise import PhenomenologicalNoise as PhenomenologicalNoise
