"""Compatibility alias for :mod:`.frontend.phase_factors`."""

import sys

from .frontend import phase_factors as _implementation
from .frontend.phase_factors import (
    capture_sparse_predicate_phase as capture_sparse_predicate_phase,
)
from .frontend.phase_factors import emit_factor as emit_factor
from .frontend.phase_factors import expand_factored_phase as expand_factored_phase

sys.modules[__name__] = _implementation
