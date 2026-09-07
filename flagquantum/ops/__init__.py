"""Internal PyTorch gate matrices and precision state."""

from __future__ import annotations

import logging
from typing import Any

from . import matrices

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = list(matrices.__all__)


def __getattr__(name: str) -> Any:
    if name == "matrices":
        return matrices
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(matrices, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
