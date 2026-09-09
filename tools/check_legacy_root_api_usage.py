#!/usr/bin/env python3
"""Reject new user-facing references to approved legacy root API names."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "contracts" / "public-api-v1-candidate.json"
TEST_DEBT = ROOT / "contracts" / "legacy-root-api-test-debt.json"
PUBLIC_FILES = (
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "flagquantum" / "ARCHITECTURE.md",
)
PUBLIC_TREES = (
    ROOT / "benchmarks",
    ROOT / "docs" / "guides",
    ROOT / "docs" / "reference",
    ROOT / "examples",
    ROOT / "flagquantum",
    ROOT / "tools",
)
EXCLUDED_NAMES = {"RELEASE_NOTES.md"}
EXCLUDED_PATHS = {
    ROOT / "benchmarks" / "mps_stability.py",
}
CHECKED_SUFFIXES = {".md", ".py", ".ipynb"}


def legacy_root_names() -> tuple[str, ...]:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    names: list[str] = list(candidate["remove_before_public"])
    for section_name in ("stable_extensions", "experimental"):
        for section in candidate[section_name]:
            names.extend(section["symbols"])
    return tuple(sorted(set(names) - set(candidate["stable_core"]["retain"])))


def violations_for_text(text: str, *, names: Iterable[str]) -> tuple[str, ...]:
    pattern = re.compile(
        rf"\bfq\.({'|'.join(re.escape(name) for name in sorted(names))})\b"
    )
    return tuple(match.group(0) for match in pattern.finditer(text))


def public_files() -> tuple[Path, ...]:
    files = list(PUBLIC_FILES)
    for tree in PUBLIC_TREES:
        files.extend(
            path
            for path in tree.rglob("*")
            if path.is_file()
            and path.suffix in CHECKED_SUFFIXES
            and path.name not in EXCLUDED_NAMES
            and path not in EXCLUDED_PATHS
        )
    return tuple(sorted(set(files)))


def validate() -> tuple[str, ...]:
    names = legacy_root_names()
    errors: list[str] = []
    for path in public_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            violations = violations_for_text(line, names=names)
            if violations:
                relative = path.relative_to(ROOT)
                errors.append(
                    f"{relative}:{line_number}: legacy root API reference "
                    f"{', '.join(violations)}"
                )
    return tuple(errors)


def validate_test_debt() -> tuple[str, ...]:
    names = legacy_root_names()
    contract = json.loads(TEST_DEBT.read_text(encoding="utf-8"))
    expected = contract["files"]
    actual: dict[str, int] = {}
    for path in (ROOT / "tests").rglob("*.py"):
        if path.name == "test_legacy_root_api_usage.py":
            continue
        count = len(violations_for_text(path.read_text(encoding="utf-8"), names=names))
        if count:
            actual[str(path.relative_to(ROOT))] = count
    errors = []
    if actual != expected:
        all_paths = sorted(set(actual) | set(expected))
        for path in all_paths:
            if actual.get(path, 0) != expected.get(path, 0):
                errors.append(
                    f"{path}: legacy root API debt changed from "
                    f"{expected.get(path, 0)} to {actual.get(path, 0)}"
                )
    actual_total = sum(actual.values())
    if actual_total != contract["total_references"]:
        errors.append(
            "legacy root API test debt total changed from "
            f"{contract['total_references']} to {actual_total}"
        )
    return tuple(errors)


def main() -> int:
    errors = (*validate(), *validate_test_debt())
    if errors:
        print("user-facing files must use approved API namespaces:")
        print("\n".join(errors))
        return 1
    print("legacy root API usage check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
