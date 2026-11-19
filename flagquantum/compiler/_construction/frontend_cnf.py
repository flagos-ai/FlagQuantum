"""Compatibility alias for :mod:`.frontend.cnf`."""

import sys

from .frontend import cnf as _implementation
from .frontend.cnf import _bitwise_condition as _bitwise_condition
from .frontend.cnf import _terms as _terms
from .frontend.cnf import capture_cnf_phase as capture_cnf_phase

sys.modules[__name__] = _implementation
