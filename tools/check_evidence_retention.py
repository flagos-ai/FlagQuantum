#!/usr/bin/env python3
"""Require an explicit lifecycle for newly added benchmark evidence.

Existing evidence is deliberately grandfathered. The gate reads the files added
by a change, then requires each new artifact under ``benchmarks/results/`` to be
registered in ``benchmarks/evidence-retention.toml``. This keeps the decision
about retaining evidence reviewable without turning repository size into a
deletion target.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path("benchmarks/evidence-retention.toml")
RESULTS_PREFIX = "benchmarks/results/"
SCHEMA = "flagquantum.benchmark_evidence_retention.v1"
RETENTION_CLASSES = frozenset({"temporary", "current_claim", "regression_baseline"})


class RetentionError(ValueError):
    """The retention manifest or changed-file input is invalid."""


def _repository_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetentionError(f"{field} must be a non-empty repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise RetentionError(f"{field} must be a normalized repository-relative path")
    return value


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RetentionError(f"{field} must be a non-empty list")
    return tuple(
        _repository_path(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def load_manifest(*, root: Path = ROOT) -> Mapping[str, object]:
    path = root / MANIFEST_PATH
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RetentionError(f"cannot read {MANIFEST_PATH}: {exc}") from exc
    return payload


def _is_evidence_file(path: str) -> bool:
    return (
        path.startswith(RESULTS_PREFIX)
        and PurePosixPath(path).name.lower() != "readme.md"
    )


def retention_errors(
    added_paths: Iterable[str],
    *,
    root: Path = ROOT,
    manifest: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return policy violations for a manifest and a set of added paths."""

    errors: list[str] = []
    if manifest is None:
        try:
            manifest = load_manifest(root=root)
        except RetentionError as exc:
            return (str(exc),)

    if manifest.get("schema") != SCHEMA:
        errors.append(f"{MANIFEST_PATH}: schema must be {SCHEMA!r}")
    entries_value = manifest.get("entries")
    if not isinstance(entries_value, dict):
        return (*errors, f"{MANIFEST_PATH}: entries must be an object")

    entries: Mapping[object, object] = entries_value
    registered: set[str] = set()
    for raw_path, raw_entry in sorted(entries.items(), key=lambda item: str(item[0])):
        try:
            path = _repository_path(raw_path, field="entry path")
        except RetentionError as exc:
            errors.append(str(exc))
            continue
        registered.add(path)
        if not _is_evidence_file(path):
            errors.append(
                f"{path}: retention entries must name a benchmark result file"
            )
        elif not (root / path).is_file():
            errors.append(f"{path}: retention entry points to a missing file")

        if not isinstance(raw_entry, dict):
            errors.append(f"{path}: retention entry must be an object")
            continue
        entry: Mapping[object, object] = raw_entry
        retention_class = entry.get("class")
        if retention_class not in RETENTION_CLASSES:
            errors.append(f"{path}: class must be one of {sorted(RETENTION_CLASSES)!r}")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{path}: reason must explain why the artifact is retained")

        if retention_class == "temporary":
            review_after = entry.get("review_after")
            try:
                if not isinstance(review_after, str):
                    raise ValueError
                date.fromisoformat(review_after)
            except ValueError:
                errors.append(
                    f"{path}: temporary evidence needs review_after as YYYY-MM-DD"
                )
        elif retention_class in {"current_claim", "regression_baseline"}:
            try:
                consumers = _string_list(
                    entry.get("consumers"), field=f"{path}.consumers"
                )
            except RetentionError as exc:
                errors.append(str(exc))
            else:
                for consumer in consumers:
                    if not (root / consumer).is_file():
                        errors.append(f"{path}: consumer does not exist: {consumer}")

        if "supersedes" in entry:
            try:
                superseded = _string_list(
                    entry["supersedes"], field=f"{path}.supersedes"
                )
            except RetentionError as exc:
                errors.append(str(exc))
            else:
                for replaced in superseded:
                    if not replaced.startswith(RESULTS_PREFIX):
                        errors.append(
                            f"{path}: supersedes must name a path under {RESULTS_PREFIX}"
                        )

    for raw_path in added_paths:
        try:
            path = _repository_path(raw_path, field="added path")
        except RetentionError as exc:
            errors.append(str(exc))
            continue
        if _is_evidence_file(path) and path not in registered:
            errors.append(
                f"{path}: new benchmark evidence needs an entry in {MANIFEST_PATH}"
            )
    return tuple(errors)


def _git_added_files(base: str, *, root: Path = ROOT) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=A", f"{base}...HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "git diff failed"
        raise RetentionError(f"cannot compare with {base!r}: {detail}") from exc
    return tuple(line for line in result.stdout.splitlines() if line)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="Git revision to compare with HEAD")
    source.add_argument(
        "--files", nargs="+", metavar="PATH", help="Explicit added paths to check"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        added_paths = _git_added_files(args.base) if args.base else tuple(args.files)
    except RetentionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = retention_errors(added_paths)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("benchmark evidence retention policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
