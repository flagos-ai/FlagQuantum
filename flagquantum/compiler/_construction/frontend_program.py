"""Compatibility alias for :mod:`.frontend.program`."""

import sys

from .frontend import program as _implementation
from .frontend.program import _Body as _Body
from .frontend.program import _Capture as _Capture
from .frontend.program import capture_program as capture_program

sys.modules[__name__] = _implementation
