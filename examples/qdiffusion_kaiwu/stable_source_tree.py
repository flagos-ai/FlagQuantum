"""Stable no-follow snapshots for reviewed QDiffusion source trees."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_STABLE_METADATA_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


@dataclass(frozen=True)
class RegularTreeSnapshot:
    root: Path
    files: frozenset[str]
    directories: frozenset[str]
    verified_files: tuple[dict[str, Any], ...]
    identities: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class RegularFileSnapshot:
    path: Path
    sha256: str
    size: int
    identity: tuple[int, ...]
    parent: Path
    parent_identity: tuple[int, ...]


def _stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(metadata, field) for field in _STABLE_METADATA_FIELDS)


def _hash_stable_file(target: Path, metadata: os.stat_result, *, label: str) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError(f"platform cannot safely hash {label}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise ValueError(f"{label} file changed before hashing: {target.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _stream_sha256(stream)
        after = os.fstat(descriptor)
        visible = target.lstat()
        if (
            _metadata_identity(after) != _metadata_identity(opened)
            or _metadata_identity(visible) != _metadata_identity(opened)
        ):
            raise ValueError(f"{label} file changed during hashing: {target.name}")
        return digest
    except OSError:
        raise ValueError(f"{label} file changed during hashing: {target.name}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def revalidate_regular_file(snapshot: RegularFileSnapshot, *, label: str) -> None:
    """Require one visible file and its parent to retain captured identities."""

    try:
        parent = snapshot.parent.lstat()
        visible = snapshot.path.lstat()
    except OSError:
        raise ValueError(f"{label} changed after identity capture") from None
    if (
        snapshot.parent.is_symlink()
        or not stat.S_ISDIR(parent.st_mode)
        or _metadata_identity(parent) != snapshot.parent_identity
        or snapshot.path.is_symlink()
        or not stat.S_ISREG(visible.st_mode)
        or _metadata_identity(visible) != snapshot.identity
    ):
        raise ValueError(f"{label} changed after identity capture")


def capture_regular_file(path: Path, *, label: str) -> RegularFileSnapshot:
    """Hash one regular file and retain its leaf and parent identities."""

    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    parent_path = path.parent
    try:
        parent = parent_path.lstat()
        metadata = path.lstat()
    except OSError:
        raise ValueError(f"{label} must be a regular, non-symlink file") from None
    if parent_path.is_symlink() or not stat.S_ISDIR(parent.st_mode):
        raise ValueError(f"{label} parent must be a regular, non-symlink directory")
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular, non-symlink file")
    snapshot = RegularFileSnapshot(
        path=path,
        sha256=_hash_stable_file(path, metadata, label=label),
        size=metadata.st_size,
        identity=_metadata_identity(metadata),
        parent=parent_path,
        parent_identity=_metadata_identity(parent),
    )
    revalidate_regular_file(snapshot, label=label)
    return snapshot


@contextmanager
def open_captured_regular_file(
    snapshot: RegularFileSnapshot, *, label: str
) -> Iterator[BinaryIO]:
    """Consume a captured file description and recheck its identity afterwards."""

    revalidate_regular_file(snapshot, label=label)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError(f"platform cannot safely open {label}")
    descriptor: int | None = None
    body_completed = False
    try:
        descriptor = os.open(
            snapshot.path,
            os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if _metadata_identity(opened) != snapshot.identity:
            raise ValueError(f"{label} changed before consumption")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield stream
        body_completed = True
    except OSError:
        raise ValueError(f"{label} changed during consumption") from None
    finally:
        try:
            if body_completed and descriptor is not None:
                if _metadata_identity(os.fstat(descriptor)) != snapshot.identity:
                    raise ValueError(f"{label} changed during consumption")
                revalidate_regular_file(snapshot, label=label)
        finally:
            if descriptor is not None:
                os.close(descriptor)


def revalidate_regular_tree(snapshot: RegularTreeSnapshot, *, label: str) -> None:
    """Require the visible tree to retain the captured path and metadata set."""

    root = snapshot.root
    try:
        paths = {
            ".": root,
            **{path: root / path for path in snapshot.identities if path != "."},
        }
        current_names = {
            ".",
            *(path.relative_to(root).as_posix() for path in root.rglob("*")),
        }
        if current_names != set(snapshot.identities):
            raise ValueError(f"{label} changed during verification")
        for relative, path in paths.items():
            if _metadata_identity(path.lstat()) != snapshot.identities[relative]:
                raise ValueError(f"{label} changed during verification")
    except OSError:
        raise ValueError(f"{label} changed during verification") from None


def capture_regular_tree(
    root: Path, *, label: str, path_prefix: str | None = None
) -> RegularTreeSnapshot:
    """Hash one regular source tree and retain identities for later revalidation."""

    if not root.is_absolute():
        raise ValueError(f"{label} root must be absolute")
    try:
        root_metadata = root.lstat()
    except OSError:
        raise ValueError(f"{label} root must be a regular, non-symlink directory") from None
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"{label} root must be a regular, non-symlink directory")
    files: set[str] = set()
    directories: set[str] = set()
    identities = {".": _metadata_identity(root_metadata)}
    verified_files: list[dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
                recorded_path = (
                    f"{path_prefix}/{relative}" if path_prefix else relative
                )
                verified_files.append(
                    {
                        "path": recorded_path,
                        "bytes": metadata.st_size,
                        "sha256": _hash_stable_file(path, metadata, label=label),
                    }
                )
            else:
                raise ValueError(
                    f"{label} contains a special filesystem entry: {relative}"
                )
            identities[relative] = _metadata_identity(metadata)
    except OSError:
        raise ValueError(f"{label} changed during verification") from None
    snapshot = RegularTreeSnapshot(
        root=root,
        files=frozenset(files),
        directories=frozenset(directories),
        verified_files=tuple(verified_files),
        identities=identities,
    )
    revalidate_regular_tree(snapshot, label=label)
    return snapshot


__all__ = (
    "RegularFileSnapshot",
    "RegularTreeSnapshot",
    "capture_regular_file",
    "capture_regular_tree",
    "open_captured_regular_file",
    "revalidate_regular_file",
    "revalidate_regular_tree",
)
