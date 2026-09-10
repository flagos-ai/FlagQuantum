"""Explicitly unstable FlagQuantum APIs with no compatibility guarantee.

Experimental capabilities are grouped by domain. Feature symbols are not
re-exported here so discovery stays small and ownership remains clear.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "artifacts",
    "distributed",
    "dynamic",
    "execution",
    "interop",
    "mps",
    "numerics",
    "planning",
    "simulation",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            "use a domain namespace such as experimental.dynamic or "
            "experimental.distributed"
        )
    return import_module(f"flagquantum.experimental.{name}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
