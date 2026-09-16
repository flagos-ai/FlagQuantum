#!/usr/bin/env python3
"""Enforce the repository coverage floor and per-package thresholds."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "contracts" / "coverage-policy.toml"
DEFAULT_XML = ROOT / "coverage.xml"
EXPECTED_SCHEMA = "flagquantum_coverage_policy_v1"


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def discover_source_packages(root: Path) -> set[str]:
    src = root / "flagquantum"
    found: set[str] = set()
    for py in src.rglob("*.py"):
        parts = py.relative_to(root).parts[:-1]
        if parts:
            found.add(".".join(parts))
    return found


def parse_coverage(path: Path) -> tuple[float, dict[str, float]]:
    root = ET.parse(path).getroot()
    global_rate = float(root.get("line-rate", "0")) * 100.0
    totals: dict[str, list[int]] = {}
    for cls in root.iter("class"):
        filename = cls.get("filename", "")
        if not filename:
            continue
        parts = Path(filename).parts
        package = ".".join(parts[:-1]) if len(parts) > 1 else filename
        lines = cls.find("lines")
        line_els = lines.findall("line") if lines is not None else []
        statements = len(line_els)
        covered = sum(1 for line in line_els if int(line.get("hits", "0")) > 0)
        bucket = totals.setdefault(package, [0, 0])
        bucket[0] += statements
        bucket[1] += covered
    packages = {
        package: 100.0 * covered / statements if statements else 100.0
        for package, (statements, covered) in totals.items()
    }
    return global_rate, packages


def policy_errors(
    policy: dict, global_rate: float, packages: dict[str, float], root: Path
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

    allowlist = set((policy.get("allowlist") or {}).keys())
    for package in sorted(discover_source_packages(root)):
        if package not in packages and package not in allowlist:
            errors.append(
                f"package {package} is absent from the coverage report and not allowlisted"
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
    errors = policy_errors(policy, global_rate, packages, ROOT)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"coverage policy passed (global {global_rate:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
