"""Fail-closed normalization for live QBoson command identifiers."""

from __future__ import annotations


def normalize_provider_identifier(value: object, *, label: str) -> str:
    """Return canonical printable text or reject it before credential access."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized or not normalized.isprintable():
        raise ValueError(f"{label} must be a non-empty printable string")
    return normalized
