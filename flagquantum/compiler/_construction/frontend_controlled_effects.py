"""Compatibility alias for :mod:`.frontend.controlled_effects`."""

import sys

from .frontend import controlled_effects as _implementation
from .frontend.controlled_effects import (
    capture_controlled_effects as capture_controlled_effects,
)

sys.modules[__name__] = _implementation
