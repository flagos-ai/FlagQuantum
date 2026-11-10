"""Frozen import compatibility for :mod:`flagquantum.realtime`.

Implementation and new consumers belong in the canonical sibling of Runtime.
Keep old submodules as aliases so imports and previously written pickles resolve
to the same classes, exceptions and enums, without loading a second copy.
"""

import sys
from importlib import import_module

from ... import realtime as _realtime

__all__ = _realtime.__all__

for _name in __all__:
    globals()[_name] = getattr(_realtime, _name)

for _name in ("protocol", "payload", "service", "channels", "session"):
    _module = import_module(f"flagquantum.realtime.{_name}")
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module
