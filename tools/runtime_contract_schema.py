#!/usr/bin/env python3
"""Generate deterministic documentation for typed runtime contracts."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from flagquantum.core.contracts import CONTRACT_TYPES, CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "runtime_contracts.schema.json"


def build_schema() -> dict[str, object]:
    return {
        "schema": "flagquantum_runtime_contracts",
        "version": CONTRACT_VERSION,
        "unknown_production_fields": "reject",
        "contracts": {
            contract.KIND: {
                "python_type": contract.__name__,
                "fields": tuple(item.name for item in fields(contract)),
            }
            for contract in CONTRACT_TYPES
        },
    }


def render_schema() -> str:
    return json.dumps(build_schema(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_schema()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit("runtime contract schema is stale; run this tool")
    else:
        OUTPUT.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
