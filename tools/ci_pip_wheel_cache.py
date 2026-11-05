"""Seed pip's private HTTP cache with validated immutable wheel responses."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def _replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".fq-wheel-", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def seed_wheel_responses(source: Path, destination: Path) -> int:
    """Copy complete wheels, leaving index and version-discovery responses out."""
    directory = source / "http-v2"
    count = 0
    for body in sorted(directory.rglob("*.body")):
        metadata = body.with_suffix("")
        if body.is_symlink() or metadata.is_symlink() or not metadata.is_file():
            continue
        with body.open("rb") as incoming:
            if incoming.read(4) != b"PK\x03\x04":
                continue
        with zipfile.ZipFile(body) as wheel:
            if not any(name.endswith(".dist-info/WHEEL") for name in wheel.namelist()):
                continue
        relative = body.relative_to(source)
        target = destination / relative
        # Pip retains its own URL, freshness, Vary, and package-hash checks.
        # Publish the complete body before its opaque HTTP metadata. Atomic
        # replacement keeps other jobs from seeing a partially copied wheel.
        _replace_file(body, target)
        _replace_file(metadata, target.with_suffix(""))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        f"Seeded {seed_wheel_responses(args.source, args.destination)} wheel responses"
    )


if __name__ == "__main__":
    main()
