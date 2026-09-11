"""Small, dependency-free contract shared by reproducible benchmark runners."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import tempfile
from pathlib import Path
from typing import Any, Mapping


def runtime_metadata(*, runner: str, schema: str, **extra: Any) -> dict[str, Any]:
    """Return stable execution metadata without probing accelerators."""
    payload: dict[str, Any] = {
        "runner": str(runner),
        "schema": str(schema),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    payload.update(extra)
    return payload


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON payload atomically and return its resolved path."""
    validate_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the runner identity fields shared by all result payloads."""
    for key in ("runner", "schema"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"benchmark payload requires non-empty string: {key}")
    if re.search(r"\.v[0-9]+$", payload["schema"]) is None:
        raise ValueError("benchmark payload schema must end with .vN")


__all__ = ["runtime_metadata", "validate_payload", "write_json_atomic"]
