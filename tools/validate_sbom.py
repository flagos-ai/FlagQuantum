#!/usr/bin/env python
"""Fail closed when a generated CycloneDX SBOM lacks release provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def sbom_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("bomFormat") != "CycloneDX":
        errors.append("SBOM must use CycloneDX")
    if not payload.get("specVersion"):
        errors.append("SBOM must declare specVersion")
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        errors.append("SBOM must contain components")
    elif not any(
        isinstance(component, dict)
        and str(component.get("name", "")).lower() == "flagquantum"
        for component in components
    ):
        errors.append("SBOM must contain the flagquantum component")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: validate_sbom.py SBOM.json", file=sys.stderr)
        return 2
    payload = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    errors = sbom_errors(payload)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("SBOM validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
