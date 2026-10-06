"""Private, durable JSON publication for QBoson acceptance evidence."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def validate_private_json_output_path(path: Path) -> None:
    """Validate an unused private JSON destination without changing the filesystem."""

    if not path.is_absolute():
        raise ValueError("evidence output path must be absolute")
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError:
        raise ValueError(
            "evidence parent must be an existing private, non-symlink directory"
        ) from None
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise ValueError(
            "evidence parent must be an existing private, non-symlink directory"
        )
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise ValueError("evidence output path could not be safely inspected") from None
    raise FileExistsError(f"evidence output path already exists: {path}")


def write_private_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON record without following or replacing unsafe paths."""

    validate_private_json_output_path(path)
    parent = path.parent

    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ("validate_private_json_output_path", "write_private_json_exclusive")
