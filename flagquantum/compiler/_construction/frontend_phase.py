"""Compatibility alias for :mod:`.frontend.phase`."""

import sys

from .frontend import phase as _implementation
from .frontend.phase import capture_polynomial_phase as capture_polynomial_phase
from .frontend.phase import expand_native_phase as expand_native_phase

sys.modules[__name__] = _implementation
