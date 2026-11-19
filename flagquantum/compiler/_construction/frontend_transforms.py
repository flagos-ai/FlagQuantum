"""Compatibility alias for :mod:`.frontend.transforms`."""

import sys

from .frontend import transforms as _implementation
from .frontend.transforms import capture_modular_exp as capture_modular_exp
from .frontend.transforms import capture_transform as capture_transform

sys.modules[__name__] = _implementation
