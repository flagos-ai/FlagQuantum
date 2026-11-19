"""Compatibility alias for :mod:`.frontend.regions`."""

import sys

from .frontend import regions as _implementation
from .frontend.regions import _checked_effects as _checked_effects
from .frontend.regions import _emit_inverse as _emit_inverse
from .frontend.regions import capture_adjoint as capture_adjoint
from .frontend.regions import capture_unitary_within as capture_unitary_within

sys.modules[__name__] = _implementation
