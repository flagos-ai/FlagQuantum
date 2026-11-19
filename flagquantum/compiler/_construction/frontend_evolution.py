"""Compatibility alias for :mod:`.frontend.evolution`."""

import sys

from .frontend import evolution as _implementation
from .frontend.evolution import _emit_rotations as _emit_rotations
from .frontend.evolution import _Evolution as _Evolution
from .frontend.evolution import _observable_terms as _observable_terms
from .frontend.evolution import _pauli_inputs as _pauli_inputs
from .frontend.evolution import capture_qdrift as capture_qdrift
from .frontend.evolution import capture_suzuki_trotter as capture_suzuki_trotter
from .frontend.evolution import flush_evolutions as flush_evolutions

sys.modules[__name__] = _implementation
