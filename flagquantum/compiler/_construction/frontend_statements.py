"""Compatibility alias for :mod:`.frontend.statements`."""

import sys

from .frontend import statements as _implementation
from .frontend.statements import _condition as _condition
from .frontend.statements import capture_conditional as capture_conditional
from .frontend.statements import capture_statements as capture_statements

sys.modules[__name__] = _implementation
