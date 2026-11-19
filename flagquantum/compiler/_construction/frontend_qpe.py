"""Compatibility alias for :mod:`.frontend.qpe`."""

import sys

from .frontend import qpe as _implementation
from .frontend.qpe import capture_qpe as capture_qpe

sys.modules[__name__] = _implementation
