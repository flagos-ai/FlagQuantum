"""Compatibility alias for :mod:`.frontend.callback_controls`."""

import sys

from .frontend import callback_controls as _implementation
from .frontend.callback_controls import capture_predicate_x as capture_predicate_x
from .frontend.callback_controls import scalar_comparison as scalar_comparison

sys.modules[__name__] = _implementation
