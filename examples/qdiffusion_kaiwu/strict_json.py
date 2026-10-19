"""Unambiguous JSON parsing for frozen QBoson evidence and configuration."""

from __future__ import annotations

import json
import math
from typing import Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate object keys")
        result[key] = value
    return result


def loads_json_strict(encoded: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting ambiguous keys and non-finite numbers."""

    value = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("JSON contains a non-finite number")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


__all__ = ("loads_json_strict",)
