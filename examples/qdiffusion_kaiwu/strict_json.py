"""Unambiguous JSON parsing for frozen QBoson evidence and configuration."""

from __future__ import annotations

import json
from typing import Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate object keys")
        result[key] = value
    return result


def loads_json_strict(encoded: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys at every nesting level."""

    return json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)


__all__ = ("loads_json_strict",)
