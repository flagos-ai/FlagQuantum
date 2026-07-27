#!/usr/bin/env python
"""Validate the single source of truth for capability maturity claims."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "capability-maturity.toml"
EXPECTED_LEVELS = (
    "experimental",
    "development_evidence",
    "production_supported",
    "release_certified",
)


def maturity_errors(data: dict[str, Any], root: Path = ROOT) -> tuple[str, ...]:
    errors: list[str] = []
    if data.get("schema") != "flagquantum_capability_maturity_v1":
        errors.append("unsupported capability maturity schema")
    levels = data.get("levels", {})
    if tuple(levels) != EXPECTED_LEVELS:
        errors.append(f"levels must be ordered as {EXPECTED_LEVELS!r}")
    ranks = [levels.get(name, {}).get("rank") for name in EXPECTED_LEVELS]
    if ranks != list(range(len(EXPECTED_LEVELS))):
        errors.append("maturity ranks must be contiguous from zero")

    capabilities = data.get("capabilities", {})
    if not capabilities:
        errors.append("at least one capability must be classified")
    for name, capability in capabilities.items():
        level = capability.get("level")
        if level not in levels:
            errors.append(f"{name}: unknown maturity level {level!r}")
            continue
        for field in levels[level].get("required_evidence", ()):
            value = capability.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name}: {level} requires {field}")
        if level != "release_certified" and capability.get("release_gate"):
            errors.append(
                f"{name}: non-certified capability cannot declare release_gate"
            )
        for field in (
            "focused_tests",
            "integration_tests",
            "operational_runbook",
            "release_gate",
            "release_artifact",
        ):
            value = capability.get(field)
            if isinstance(value, str) and value != "not_applicable_pure_contract":
                if not (root / value).exists():
                    errors.append(f"{name}: {field} path does not exist: {value}")
    return tuple(errors)


def main() -> int:
    data = tomllib.loads(MATRIX.read_text(encoding="utf-8"))
    errors = maturity_errors(data)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("capability maturity matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
