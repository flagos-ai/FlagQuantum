#!/usr/bin/env python3
"""Reject generated, oversized, dataset and model artifacts in Git."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".parquet",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
}
ALLOWED_LARGE_FILES: dict[Path, int] = {}
DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_MAX_TOTAL_BYTES = 50_000_000


def tracked_files(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=root,
        capture_output=True,
    )
    if completed.returncode != 0:
        git_dir = root
        while git_dir != git_dir.parent and not (git_dir / ".git").exists():
            git_dir = git_dir.parent
        metadata = git_dir / ".git"
        if not metadata.exists() or (
            metadata.is_dir() and not (metadata / "HEAD").exists()
        ):
            return ()
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    output = completed.stdout
    return tuple(root / item.decode() for item in output.split(b"\0") if item)


def violations(
    root: Path, *, max_file_bytes: int, max_total_bytes: int
) -> tuple[str, ...]:
    errors = []
    total = 0
    for path in tracked_files(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        size = path.stat().st_size
        total += size
        if FORBIDDEN_PARTS.intersection(relative.parts):
            errors.append(f"generated cache tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data artifact tracked: {relative}")
        allowed_size = ALLOWED_LARGE_FILES.get(relative, max_file_bytes)
        if size > allowed_size:
            errors.append(f"tracked file exceeds {max_file_bytes} bytes: {relative}")
        if path.read_bytes()[:43] == b"version https://git-lfs.github.com/spec/v1":
            errors.append(f"Git-LFS pointer is not allowed in source: {relative}")
    if total > max_total_bytes:
        errors.append(f"tracked tree is {total} bytes; limit is {max_total_bytes}")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = violations(
        root,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    if errors:
        print("\n".join(errors))
        return 1
    print("repository hygiene passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
