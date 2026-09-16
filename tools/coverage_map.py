#!/usr/bin/env python3
"""Print per-package (or per-module) line coverage, worst first."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "coverage.json"


def _dotted(path: str, *, module: bool) -> str:
    if module:
        return ".".join(Path(path).with_suffix("").parts)
    return ".".join(Path(path).parts[:-1])


def build_map(data: dict, *, module: bool) -> list[tuple[str, int, int, float]]:
    totals: dict[str, list[int]] = {}
    for path, info in (data.get("files") or {}).items():
        if not path.startswith("flagquantum"):
            continue
        summary = info.get("summary") or {}
        statements = int(summary.get("num_statements", 0))
        missing = int(summary.get("missing_lines", 0))
        bucket = totals.setdefault(_dotted(path, module=module), [0, 0])
        bucket[0] += statements
        bucket[1] += missing
    rows: list[tuple[str, int, int, float]] = []
    for key, (statements, missing) in totals.items():
        percent = 100.0 * (statements - missing) / statements if statements else 100.0
        rows.append((key, statements, missing, percent))
    rows.sort(key=lambda row: (row[3], row[0]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--min", type=float, default=0.0, help="only show entries below this percentage"
    )
    parser.add_argument("--by-module", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(
            f"coverage.json not found at {args.input}; run pytest with --cov-report=json first"
        )
        return 1
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for key, statements, missing, percent in build_map(data, module=args.by_module):
        if percent < args.min:
            print(f"{percent:5.1f}%  {statements - missing:4d}/{statements}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
