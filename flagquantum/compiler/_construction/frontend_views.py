"""Compatibility alias for :mod:`.frontend.views`."""

import sys

from .frontend import views as _implementation
from .frontend.views import _require_builtin as _require_builtin
from .frontend.views import capture_loop as capture_loop
from .frontend.views import expand_loop as expand_loop
from .frontend.views import gate_target as gate_target
from .frontend.views import inline_mutable as inline_mutable
from .frontend.views import locations as locations
from .frontend.views import packed_locations as packed_locations
from .frontend.views import static_integer as static_integer

sys.modules[__name__] = _implementation
