"""Compatibility alias for :mod:`.frontend.repeated_effects`."""

import sys

from .frontend import repeated_effects as _implementation
from .frontend.repeated_effects import _contains_reset as _contains_reset
from .frontend.repeated_effects import capture_reset_loop as capture_reset_loop

sys.modules[__name__] = _implementation
