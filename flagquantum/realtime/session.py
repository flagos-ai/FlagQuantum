"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.realtime.session import (
    RealtimeSession as RealtimeSession,
)
from flagquantum.compute.realtime.session import __all__ as __all__

_implementation = _import_module("flagquantum.compute.realtime.session")
_sys.modules[__name__] = _implementation
