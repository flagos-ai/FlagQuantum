"""Compatibility alias for :mod:`.frontend.fixed_xor`."""

import sys

from .frontend import fixed_xor as _implementation
from .frontend.fixed_xor import capture_fixed_xor as capture_fixed_xor
from .frontend.fixed_xor import is_fixed_xor_helper as is_fixed_xor_helper

sys.modules[__name__] = _implementation
