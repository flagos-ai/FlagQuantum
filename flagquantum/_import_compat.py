"""Internal import identity support for the structure migration."""

from __future__ import annotations

import sys
from types import FunctionType


def alias_module(previous: str, current: str) -> None:
    """Share module state and bind a loaded historical parent for old imports."""
    if current != "flagquantum" and not current.startswith("flagquantum."):
        return
    module = sys.modules.get(current)
    if module is None:
        return
    for value in vars(module).values():
        if isinstance(value, (type, FunctionType)) and value.__module__ == current:
            value.__module__ = previous
    sys.modules[previous] = module
    parent_name, _, child = previous.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, module)
