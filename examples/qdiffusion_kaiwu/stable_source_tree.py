"""Stable no-follow snapshots for reviewed QDiffusion source trees."""

from __future__ import annotations

import hashlib
import os
import stat
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
    "RegularTreeSnapshot",
    "capture_regular_tree",
    "revalidate_regular_tree",
)
