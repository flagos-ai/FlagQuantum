"""Compatibility alias for :mod:`.frontend.xor`."""

import sys

from .frontend import xor as _implementation
from .frontend.xor import _predicate_expression as _predicate_expression
from .frontend.xor import capture_expression_xor as capture_expression_xor

sys.modules[__name__] = _implementation
