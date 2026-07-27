#!/usr/bin/env python3
"""Generate the executable operator and backend capability manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flagquantum.core.operator_schema import operator_manifest
from flagquantum.ops.lowering import DEFAULT_LOWERING_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "operator_manifest.json"


def build_manifest() -> dict[str, object]:
    return {
        "schema_version": "flagquantum_operator_manifest_v1",
        "operators": operator_manifest(),
        "lowerings": DEFAULT_LOWERING_REGISTRY.manifest()["backends"],
    }


def render_manifest() -> str:
    return json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rendered = render_manifest()
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"operator manifest is stale: run {Path(__file__).name}")
        return 0
    args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
