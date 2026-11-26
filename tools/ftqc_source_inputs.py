"""Materialize immutable FTQC test sources without installing runtime packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

FTQC_SOURCE_INVENTORY = (
    Path(__file__).resolve().parents[1] / "docs/development/ftqc_source_inventory.csv"
)
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class SourceFile:
    identifier: str
    revision: str
    path: str
    git_blob: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.identifier, self.revision, self.path, self.git_blob)
        ):
            raise ValueError("Source inventory fields must be strings")
        if not re.fullmatch(r"[QK]-[0-9]{4}", self.identifier):
            raise ValueError(f"Invalid source identifier: {self.identifier!r}")
        if not all(
            re.fullmatch(r"[0-9a-f]{40}", value)
            for value in (self.revision, self.git_blob)
        ):
            raise ValueError(f"Invalid source revision or blob: {self.identifier}")
        relative = PurePosixPath(self.path)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or str(relative) != self.path
            or "\\" in self.path
            or re.match(r"^[A-Za-z]:", self.path)
        ):
            raise ValueError(f"Invalid source path: {self.path!r}")


def read_inventory(path: Path = FTQC_SOURCE_INVENTORY) -> tuple[SourceFile, ...]:
    """Read the existing native inventory; availability labels do not filter it."""
    records = []
    identifiers: set[str] = set()
    paths: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "revision", "path", "git_blob"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Source inventory is missing required columns")
        for row in reader:
            identifier, revision, name, blob = (
                row[key] for key in ("id", "revision", "path", "git_blob")
            )
            record = SourceFile(identifier, revision, name, blob)
            if identifier in identifiers or name in paths:
                raise ValueError(f"Duplicate source identifier or path: {identifier}")
            identifiers.add(identifier)
            paths.add(name)
            records.append(record)
    if not records:
        raise ValueError("Source inventory is empty")
    return tuple(records)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_sources(root: Path, records: Sequence[SourceFile]) -> None:
    """Require every declared source to have its exact Git blob."""
    if not records:
        raise ValueError("Source inventory is empty")
    resolved = root.resolve()
    for record in records:
        path = root / record.path
        if not path.resolve().is_relative_to(resolved):
            raise ValueError(f"Source escapes its root: {record.path}")
        if not path.is_file() or git_blob(path.read_bytes()) != record.git_blob:
            raise ValueError(
                f"Missing or changed frozen source: {record.identifier} {record.path}"
            )


def _copy_archive(
    archive_path: Path, root: Path, records: Sequence[SourceFile], budget: int
) -> int:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        prefixes = {
            PurePosixPath(member.name).parts[0] for member in members if member.name
        }
        if len(prefixes) != 1:
            raise ValueError("Source archive must have one repository root")
        prefix = next(iter(prefixes))
        names = [member.name for member in members]
        if len(set(names)) != len(names):
            raise ValueError("Source archive has duplicate members")
        try:
            selected = [
                archive.getmember(f"{prefix}/{record.path}") for record in records
            ]
        except KeyError as error:
            raise ValueError(f"Missing frozen archive source: {error}") from error
        if any(not member.isfile() for member in selected):
            raise ValueError("Source inventory requires regular archive files")
        total = sum(member.size for member in selected)
        if total > budget:
            raise ValueError("Source snapshot exceeds the size budget")
        for record, member in zip(records, selected, strict=True):
            if member.size > _MAX_ARCHIVE_BYTES:
                raise ValueError(f"Source file exceeds the size budget: {record.path}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read source: {record.path}")
            with stream:
                data = stream.read(_MAX_ARCHIVE_BYTES + 1)
            if len(data) != member.size or git_blob(data) != record.git_blob:
                raise ValueError(
                    f"Changed frozen archive source: {record.identifier} {record.path}"
                )
            target = root / record.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
        return total


def prepare_sources(
    archives: Mapping[str, Path],
    destination: Path,
    records: Sequence[SourceFile],
) -> None:
    """Publish a fully verified snapshot, leaving existing trees unchanged."""
    if not records:
        raise ValueError("Source inventory is empty")
    if destination.exists() or destination.is_symlink():
        verify_sources(destination, records)
        return
    revisions = {record.revision for record in records}
    if set(archives) != revisions:
        raise ValueError("Archives do not match the inventory revisions")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fq-sources-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "snapshot"
        staged.mkdir()
        remaining = _MAX_SOURCE_BYTES
        for revision in sorted(revisions):
            remaining -= _copy_archive(
                archives[revision],
                staged,
                [r for r in records if r.revision == revision],
                remaining,
            )
        verify_sources(staged, records)
        needed = _MAX_SOURCE_BYTES - remaining
        if shutil.disk_usage(destination.parent).free < needed + 1024**3:
            raise OSError("Source preparation must leave at least 1 GiB free")
        # Exclusive creation also protects a tree created after verification.
        destination.mkdir()
        try:
            shutil.copytree(staged, destination, dirs_exist_ok=True)
            verify_sources(destination, records)
        except BaseException:
            shutil.rmtree(destination)
            raise


def download_archive(revision: str, cache: Path) -> Path:
    """Download public source bytes with bounded size/time and no credentials."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full Git revision is required")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{revision}.tar.gz"
    if path.is_file():
        return path
    url = f"https://codeload.github.com/quantumlib/Qualtran/tar.gz/{revision}"
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream, urlopen(url, timeout=60) as response:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_ARCHIVE_BYTES:
                    raise ValueError("Source archive exceeds the download budget")
                stream.write(chunk)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=FTQC_SOURCE_INVENTORY)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    records = read_inventory(args.inventory)
    if args.destination.exists() or args.destination.is_symlink():
        verify_sources(args.destination, records)
    else:
        archives = {
            revision: download_archive(revision, args.archive_cache)
            for revision in sorted({r.revision for r in records})
        }
        prepare_sources(archives, args.destination, records)
    print(f"Verified all {len(records)} frozen source files at {args.destination}")


if __name__ == "__main__":
    main()
