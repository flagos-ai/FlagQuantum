"""Compatibility alias for :mod:`.frontend.borrowed_expression`."""

import sys

from .frontend import borrowed_expression as _implementation
from .frontend.borrowed_expression import borrowed_expression as borrowed_expression

sys.modules[__name__] = _implementation
