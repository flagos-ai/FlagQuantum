#!/usr/bin/env python3
"""Detect or redact infrastructure identifiers in public evidence files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EVIDENCE_ROOTS = (ROOT / "artifacts", ROOT / "benchmarks" / "results")

HOSTNAME = re.compile(r'("hostname"\s*:\s*)"(?!redacted")[^"]*"')
PRIVATE_IPV4 = re.compile(
    r"(?<!\d)(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)"
    r"|(?<!\d)192\.168\.\d{1,3}\.\d{1,3}(?!\d)"
    r"|(?<!\d)172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}(?!\d)"
)
PROJECT_HOME_PREFIX = re.compile(
    r"/(?:root|home/[^/\s\"]+|Users/[^/\s\"]+)/"
    r"(?:[^/\s\"]+/)*FlagQuantum/FlagQuantum/"
)
ABSOLUTE_HOME_TOKEN = re.compile(
    r"/(?:root|home/[^/\s\"]+|Users/[^/\s\"]+)/[^\s\"]+"
)


def sanitize_text(text: str) -> str:
    """Return a deterministic public representation without local identities."""

    text = HOSTNAME.sub(r'\1"redacted"', text)
    text = PRIVATE_IPV4.sub("<redacted-private-address>", text)
    text = PROJECT_HOME_PREFIX.sub("", text)
    return ABSOLUTE_HOME_TOKEN.sub("<redacted-path>", text)


def evidence_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    for root in PUBLIC_EVIDENCE_ROOTS:
        if root.exists():
            paths.extend(path for path in root.rglob("*.json") if path.is_file())
    return tuple(sorted(paths))


def unsanitized_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in evidence_files()
        if sanitize_text(path.read_text(encoding="utf-8"))
        != path.read_text(encoding="utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite public evidence in place; the default only checks",
    )
    args = parser.parse_args()

    dirty = unsanitized_files()
    if args.write:
        for path in dirty:
            original = path.read_text(encoding="utf-8")
            path.write_text(sanitize_text(original), encoding="utf-8")
        print(f"sanitized {len(dirty)} public evidence files")
        return 0

    if dirty:
        for path in dirty:
            print(f"unsanitized public evidence: {path.relative_to(ROOT)}")
        return 1
    print("public evidence privacy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
