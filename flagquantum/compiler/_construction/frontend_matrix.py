"""Compatibility alias for :mod:`.frontend.matrix`."""

import sys

from .frontend import matrix as _implementation
from .frontend.matrix import capture_controlled_unitary as capture_controlled_unitary
from .frontend.matrix import capture_unitary as capture_unitary

sys.modules[__name__] = _implementation
