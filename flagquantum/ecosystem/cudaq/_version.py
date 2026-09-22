"""CUDA-Q package version discovery."""

from __future__ import annotations

from importlib import metadata
from typing import Any


def installed_cudaq_version(cudaq: Any) -> str:
    """Return the installed distribution version with a module fallback."""

    try:
        return metadata.version("cudaq")
    except metadata.PackageNotFoundError:
        return str(getattr(cudaq, "__version__", "unknown"))


__all__ = ("installed_cudaq_version",)
