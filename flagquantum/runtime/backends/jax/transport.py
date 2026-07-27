"""Independent JAX distributed initialization and transport facade."""

from __future__ import annotations

from typing import Any

_EXPORTS = {"initialize_jax_distributed"}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    from ._loader import resolve

    return resolve(name, domain="transport")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = sorted(_EXPORTS)
