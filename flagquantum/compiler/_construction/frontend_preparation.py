"""Compatibility alias for :mod:`.frontend.preparation`."""

import sys

from .frontend import preparation as _implementation
from .frontend.preparation import _number as _number
from .frontend.preparation import capture_preparation as capture_preparation

sys.modules[__name__] = _implementation
