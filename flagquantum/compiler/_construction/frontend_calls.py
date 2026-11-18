"""Compatibility alias for :mod:`.frontend.calls`."""

import sys

from .frontend import calls as _implementation
from .frontend.calls import capture_helper_call as capture_helper_call

sys.modules[__name__] = _implementation
