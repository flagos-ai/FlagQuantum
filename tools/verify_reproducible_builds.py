#!/usr/bin/env python3
"""Verify two isolated build directories contain byte-identical artifacts."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def _digest(path: Path) -> str:
    if path.name.endswith(".tar.gz"):
        digest = hashlib.sha256()
        with tarfile.open(path, "r:gz") as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                digest.update(
                    f"{member.name}\0{member.mode}\0{member.size}\0{member.type}".encode()
                )
                stream = archive.extractfile(member) if member.isfile() else None
                if stream is not None:
                    digest.update(stream.read())
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left = {path.name: _digest(path) for path in args.left.iterdir() if path.is_file()}
    right = {
        path.name: _digest(path) for path in args.right.iterdir() if path.is_file()
    }
    if left != right:
        print(f"non-reproducible builds:\nleft={left}\nright={right}")
        return 1
    print(f"reproducible build passed: {left}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
