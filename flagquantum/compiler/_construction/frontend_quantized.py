"""Compatibility alias for :mod:`.frontend.quantized`."""

import sys

from .frontend import quantized as _implementation
from .frontend.quantized import _emit as _emit
from .frontend.quantized import _Native as _Native
from .frontend.quantized import _reachable_marker as _reachable_marker
from .frontend.quantized import try_capture_quantized as try_capture_quantized

sys.modules[__name__] = _implementation
