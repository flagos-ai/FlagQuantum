#!/usr/bin/env python3
"""Enforce the repository coverage floor and per-package thresholds."""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import ast
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tools.architecture.source_relocations import source_identity

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "contracts" / "coverage-policy.toml"
DEFAULT_XML = ROOT / "coverage.xml"
EXPECTED_SCHEMA = "flagquantum_coverage_policy_v1"


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _package_name(source: str) -> str:
    parts = Path(source).parent.parts
    if not parts or parts[0] != "flagquantum":
        raise ValueError(f"coverage source is outside flagquantum: {source}")
    return ".".join(parts[1:]) or "."


def _line_scopes(
    source: Path, repository: Path
) -> tuple[str, list[tuple[int, int, str]]]:
    """Keep measured implementations under their original policy directories."""
    relative = source.relative_to(repository).as_posix()
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
    package_root = repository / "flagquantum"
    default = _package_name(source_identity(package_root, relative, "<module>"))
    scopes: list[tuple[int, int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = _package_name(source_identity(package_root, relative, node.name))
            start = (
                min(node.lineno, *(item.lineno for item in node.decorator_list))
                if node.decorator_list
                else node.lineno
            )
            if node.end_lineno is None:
                raise ValueError(
                    f"missing coverage source span: {relative}:{node.lineno}"
                )
            scopes.append((start, node.end_lineno, owner))
    # A partial extraction or a merge into an existing module must not donate
    # unrelated covered globals to the moved implementation's policy scope.
    if any(owner != default for _, _, owner in scopes):
        default = _package_name(relative)
    return default, scopes


def parse_coverage(
    path: Path, repository: Path = ROOT
) -> tuple[float, dict[str, float]]:
    """Read actual hits without changing global coverage or frozen scope floors.

    Whole-file and definition-level moves use the existing source ledger. Every
    measured statement is counted once, including retained public facades and
    newly introduced code. XML package summaries alone cannot describe a split
    or merged implementation and are not used as substitute measurements.
    """
    document = ET.parse(path).getroot()
    global_rate = float(document.get("line-rate", "0")) * 100.0
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    seen: set[tuple[str, int]] = set()
    for entry in document.iter("class"):
        relative = Path(entry.get("filename", "").replace("\\", "/"))
        if relative.is_absolute():
            relative = relative.relative_to(repository)
        if ".." in relative.parts:
            raise ValueError(f"invalid measured source path: {relative}")
        if not relative.parts or relative.parts[0] != "flagquantum":
            relative = Path("flagquantum") / relative
        source = repository / relative
        default, scopes = _line_scopes(source, repository)
        for line in entry.findall("./lines/line"):
            number = int(line.get("number", "0"))
            hits = int(line.get("hits", "0"))
            identity = (relative.as_posix(), number)
            if number < 1 or hits < 0 or identity in seen:
                raise ValueError(f"invalid or duplicate measured line: {identity}")
            seen.add(identity)
            owner = next(
                (owner for start, end, owner in scopes if start <= number <= end),
                default,
            )
            counts[owner][0] += int(hits > 0)
            counts[owner][1] += 1
    if not seen:
        raise ValueError("coverage XML contains no measured source lines")
    return global_rate, {
        package: 100.0 * covered / total for package, (covered, total) in counts.items()
    }


def policy_errors(
    policy: dict, global_rate: float, packages: dict[str, float]
) -> list[str]:
    errors: list[str] = []
    if policy.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"coverage policy schema must be {EXPECTED_SCHEMA}")

    global_cfg = policy.get("global")
    if not isinstance(global_cfg, dict):
        errors.append("coverage policy is missing [global]")
    else:
        floor = global_cfg.get("min")
        if not isinstance(floor, (int, float)):
            errors.append("[global].min must be a number")
        elif global_rate < float(floor):
            errors.append(f"global coverage {global_rate:.1f}% is below floor {floor}%")

    for package, cfg in (policy.get("packages") or {}).items():
        floor = cfg.get("min") if isinstance(cfg, dict) else None
        if not isinstance(floor, (int, float)):
            errors.append(f"[packages.{package}].min must be a number")
            continue
        rate = packages.get(package, 0.0)
        if rate < float(floor):
            errors.append(
                f"package {package} coverage {rate:.1f}% is below floor {floor}%"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--coverage-xml", type=Path, default=DEFAULT_XML)
    args = parser.parse_args(argv)

    if not args.coverage_xml.exists():
        print(
            f"coverage.xml not found at {args.coverage_xml}; "
            "run pytest with --cov-report=xml first"
        )
        return 1
    policy = load_toml(args.policy)
    global_rate, packages = parse_coverage(args.coverage_xml)
    errors = policy_errors(policy, global_rate, packages)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"coverage policy passed (global {global_rate:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
