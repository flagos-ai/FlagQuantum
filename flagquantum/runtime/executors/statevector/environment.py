"""Statevector runtime environment configuration."""

from __future__ import annotations

import os


def get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def mode() -> str:
    value = os.getenv("FQ_SV_RUNTIME_MODE", "auto").strip().lower()
    if value not in {"auto", "portable", "max_perf"}:
        raise ValueError("FQ_SV_RUNTIME_MODE must be auto, portable, or max_perf")
    return value


__all__ = ("get_bool", "mode")
