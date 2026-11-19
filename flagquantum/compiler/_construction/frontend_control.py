"""Compatibility alias for :mod:`.frontend.control`."""

import sys

from .frontend import control as _implementation
from .frontend.control import capture_control as capture_control
from .frontend.control import capture_controlled_x as capture_controlled_x
from .frontend.control import is_composite_predicate as is_composite_predicate

sys.modules[__name__] = _implementation
