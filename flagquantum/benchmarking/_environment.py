"""Scoped environment overrides for in-process benchmark A/B engines."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


def cpu_model() -> str:
    """Return a stable CPU model label without invoking a platform command."""

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_affinity() -> tuple[int, ...] | None:
    """Return the logical CPUs available to this process when supported."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    return tuple(sorted(int(cpu) for cpu in get_affinity(0)))


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


__all__ = ("cpu_affinity", "cpu_model", "temporary_boolean_environment")
