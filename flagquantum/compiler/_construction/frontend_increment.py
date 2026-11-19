"""Compatibility alias for :mod:`.frontend.increment`."""

import sys

from .frontend import increment as _implementation
from .frontend.increment import _emit as _emit
from .frontend.increment import _target as _target
from .frontend.increment import capture_increment as capture_increment
from .frontend.increment import capture_increment_call as capture_increment_call
from .frontend.increment import is_increment_helper as is_increment_helper

sys.modules[__name__] = _implementation
