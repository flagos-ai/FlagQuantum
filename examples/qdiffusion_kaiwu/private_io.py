"""Private, durable JSON publication for QBoson acceptance evidence."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise ValueError("platform cannot validate private evidence ownership")
    return int(getter())


def validate_private_directory(path: Path, *, label: str) -> None:
    """Validate an existing absolute private directory without following its leaf."""

    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(
            f"{label} must be an existing private, non-symlink directory"
        ) from None
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_uid != _effective_uid()
    ):
        raise ValueError(f"{label} must be an existing private, non-symlink directory")


def validate_private_json_output_path(path: Path) -> None:
    """Validate an unused private JSON destination without changing the filesystem."""

    if not path.is_absolute():
        raise ValueError("evidence output path must be absolute")
    parent = path.parent
    validate_private_directory(parent, label="evidence parent")
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise ValueError("evidence output path could not be safely inspected") from None
    raise FileExistsError(f"evidence output path already exists: {path}")


def _verify_open_directory_binding(
    parent: Path, descriptor_metadata: os.stat_result
) -> None:
    """Require the visible parent path to retain the opened private directory."""

    try:
        visible = parent.lstat()
    except OSError:
        raise ValueError("evidence parent changed during publication") from None
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(visible.st_mode)
        or visible.st_mode & 0o077
        or visible.st_uid != _effective_uid()
        or descriptor_metadata.st_uid != _effective_uid()
        or (visible.st_dev, visible.st_ino)
        != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    ):
        raise ValueError("evidence parent changed during publication")


@contextmanager
def open_private_binary(
    path: Path, *, label: str, max_bytes: int
) -> Iterator[BinaryIO]:
    """Open one bounded private file and recheck its binding after consumption."""

    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError(f"platform cannot safely read {label}")
    parent = path.parent
    validate_private_directory(parent, label=f"{label} parent")
    try:
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | directory_flag | no_follow | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError:
        raise ValueError(f"{label} parent changed during validation") from None
    descriptor: int | None = None
    body_started = False
    body_completed = False
    try:
        directory_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_mode & 0o077
            or directory_metadata.st_uid != _effective_uid()
        ):
            raise ValueError(f"{label} parent changed during validation")
        _verify_open_directory_binding(parent, directory_metadata)
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
        except OSError:
            raise ValueError(
                f"{label} must be a regular, non-symlink file"
            ) from None
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular, non-symlink file")
        if opened.st_mode & 0o077:
            raise ValueError(f"{label} must not be accessible by group or others")
        if opened.st_uid != _effective_uid():
            raise ValueError(f"{label} must be owned by the current effective user")
        if opened.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the bounded size")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            body_started = True
            yield stream
        body_completed = True
    except OSError:
        if body_started:
            raise
        raise ValueError(f"{label} binding changed during validation") from None
    finally:
        try:
            if body_completed and descriptor is not None:
                try:
                    after = os.fstat(descriptor)
                    stable_fields = (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_uid",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                    if any(
                        getattr(after, field) != getattr(opened, field)
                        for field in stable_fields
                    ):
                        raise ValueError(f"{label} changed during validation")
                    visible = path.lstat()
                    if (
                        not stat.S_ISREG(visible.st_mode)
                        or visible.st_mode & 0o077
                        or visible.st_uid != _effective_uid()
                        or after.st_uid != _effective_uid()
                        or (visible.st_dev, visible.st_ino)
                        != (opened.st_dev, opened.st_ino)
                    ):
                        raise ValueError(f"{label} binding changed during validation")
                    _verify_open_directory_binding(parent, directory_metadata)
                except OSError:
                    raise ValueError(
                        f"{label} binding changed during validation"
                    ) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory_descriptor)


def read_private_bytes(path: Path, *, label: str, max_bytes: int) -> bytes:
    """Read one bounded private file through an anchored parent descriptor."""

    with open_private_binary(path, label=label, max_bytes=max_bytes) as stream:
        encoded = stream.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            raise ValueError(f"{label} exceeds the bounded size")
        return encoded


def write_private_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON record without following or replacing unsafe paths."""

    validate_private_json_output_path(path)
    parent = path.parent

    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_descriptor = os.open(parent, directory_flags)
    except OSError:
        raise ValueError("evidence parent changed during publication") from None
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    temporary_descriptor: int | None = None
    published = False
    try:
        directory_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_mode & 0o077
            or directory_metadata.st_uid != _effective_uid()
        ):
            raise ValueError("evidence parent changed during publication")
        _verify_open_directory_binding(parent, directory_metadata)
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(temporary_descriptor, "wb") as stream:
            temporary_descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_open_directory_binding(parent, directory_metadata)
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        published = True
        _verify_open_directory_binding(parent, directory_metadata)
        final_directory_metadata = os.fstat(directory_descriptor)
        if (
            final_directory_metadata.st_mode & 0o077
            or final_directory_metadata.st_uid != _effective_uid()
        ):
            raise ValueError("evidence parent changed during publication")
        os.fsync(directory_descriptor)
        published = False
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if published:
            try:
                os.unlink(path.name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory_descriptor)


__all__ = (
    "open_private_binary",
    "read_private_bytes",
    "validate_private_directory",
    "validate_private_json_output_path",
    "write_private_json_exclusive",
)
