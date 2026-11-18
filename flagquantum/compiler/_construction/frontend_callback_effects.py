"""Compatibility alias for :mod:`.frontend.callback_effects`."""

import sys

from .frontend import callback_effects as _implementation
from .frontend.callback_effects import emit_callback_effects as emit_callback_effects

sys.modules[__name__] = _implementation
