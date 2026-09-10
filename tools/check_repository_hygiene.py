#!/usr/bin/env python3
"""Reject generated, oversized, dataset and model artifacts in Git."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from tools.sanitize_public_evidence import unsanitized_files
except ModuleNotFoundError:  # direct script execution
    from sanitize_public_evidence import unsanitized_files

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
FORBIDDEN_PAPER_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
}
ALLOWED_LARGE_FILES: dict[Path, int] = {}
DEFAULT_MAX_FILE_BYTES = 2_000_000
# The integrated MPS/statevector evidence corpus is part of the reproducibility
# contract and is referenced by benchmark-contract tests and public reports.
# Keep a bounded repository-wide budget while retaining the stricter per-file
# and forbidden-artifact checks above.
DEFAULT_MAX_TOTAL_BYTES = 110_000_000
CANONICAL_RESULT_DIRECTORIES = {"comparison", "local", "scalability", "smoke"}
ARTIFACT_CONTAINER_DIRECTORIES = {"development"}
PACKAGE_ROOT_FILES = {
    "__init__.py",
    "_api.py",
    "circuit.py",
    "dynamic.py",
    "errors.py",
    "gradients.py",
    "models.py",
    "operators.py",
    "training.py",
    "version.py",
}


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
        if relative.parts[0] == "paper" and (
            path.suffix.lower() in FORBIDDEN_PAPER_SUFFIXES
            or path.name.endswith("Notes.bib")
        ):
            errors.append(f"generated paper build artifact tracked: {relative}")
        allowed_size = ALLOWED_LARGE_FILES.get(relative, max_file_bytes)
        if size > allowed_size:
            errors.append(f"tracked file exceeds {max_file_bytes} bytes: {relative}")
        if path.read_bytes()[:43] == b"version https://git-lfs.github.com/spec/v1":
            errors.append(f"Git-LFS pointer is not allowed in source: {relative}")
    if total > max_total_bytes:
        errors.append(f"tracked tree is {total} bytes; limit is {max_total_bytes}")
    return tuple(errors)


def layout_violations(root: Path) -> tuple[str, ...]:
    errors = [
        f"capability contract must live under contracts/: {path.name}"
        for path in sorted(root.glob("*-contract.toml"))
    ]
    result_directories = {
        path.name
        for path in (root / "benchmarks" / "results").iterdir()
        if path.is_dir()
    }
    for name in sorted(result_directories - CANONICAL_RESULT_DIRECTORIES):
        errors.append(f"new top-level benchmark result family is forbidden: {name}")
    artifact_directories = {
        path.name for path in (root / "artifacts").iterdir() if path.is_dir()
    }
    for name in sorted(artifact_directories - ARTIFACT_CONTAINER_DIRECTORIES):
        errors.append(f"new top-level artifact family is forbidden: {name}")
    forbidden_roots = (
        root / ".codex" / "skills",
        root / "docs" / "reviews",
        root / "paper",
        root / "artifacts" / "legacy",
        root / "benchmarks" / "results" / "legacy",
        root / "benchmarks" / "internal" / "data",
        root / "benchmarks" / "internal" / "scripts",
        root / "benchmarks" / "development",
        root / "benchmarks" / "research",
    )
    tracked = tuple(path.relative_to(root) for path in tracked_files(root))
    package_root_files = {
        path.name
        for path in tracked
        if path.parent == Path("flagquantum") and (root / path).is_file()
    }
    for name in sorted(package_root_files - PACKAGE_ROOT_FILES):
        errors.append(f"unexpected file at package root: flagquantum/{name}")
    for path in forbidden_roots:
        relative = path.relative_to(root)
        if any(item == relative or relative in item.parents for item in tracked):
            errors.append(
                f"internal or archived asset must not be tracked here: {relative}"
            )
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = (
        violations(
            root,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )
        + layout_violations(root)
        + tuple(
            f"unsanitized public evidence: {path.relative_to(root)}"
            for path in unsanitized_files()
        )
    )
    if errors:
        print("\n".join(errors))
        return 1
    print("repository hygiene passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
