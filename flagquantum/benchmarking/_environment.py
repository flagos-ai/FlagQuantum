"""Scoped environment overrides for in-process benchmark A/B engines."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


@contextmanager
def temporary_boolean_environment(
    overrides: Mapping[str, bool | None],
) -> Iterator[None]:
    """Apply boolean feature switches and restore their exact prior values."""

    active = {
        name: enabled for name, enabled in overrides.items() if enabled is not None
    }
    prior = {name: os.environ.get(name) for name in active}
    try:
        for name, enabled in active.items():
            os.environ[name] = "1" if enabled else "0"
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
