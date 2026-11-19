"""Compatibility alias for :mod:`.frontend.classical`."""

import sys

from .frontend import classical as _implementation
from .frontend.classical import ClassicalValue as ClassicalValue
from .frontend.classical import _checked_integer as _checked_integer
from .frontend.classical import _checked_real as _checked_real
from .frontend.classical import bind_helper as bind_helper
from .frontend.classical import bind_inputs as bind_inputs
from .frontend.classical import classical_number as classical_number
from .frontend.classical import real_number as real_number

sys.modules[__name__] = _implementation
