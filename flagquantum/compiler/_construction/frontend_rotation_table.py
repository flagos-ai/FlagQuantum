"""Compatibility alias for :mod:`.frontend.rotation_table`."""

import sys

from .frontend import rotation_table as _implementation
from .frontend.rotation_table import capture_rotation_table as capture_rotation_table

sys.modules[__name__] = _implementation
