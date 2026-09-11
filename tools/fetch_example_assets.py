#!/usr/bin/env python3
"""Fetch versioned example assets and verify size and SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "examples" / "assets" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "examples" / ".assets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(path: Path, asset: dict[str, Any]) -> None:
    if path.stat().st_size != int(asset["bytes"]):
        raise ValueError(f"asset size mismatch: {asset['id']}")
    if _sha256(path) != asset["sha256"]:
        raise ValueError(f"asset checksum mismatch: {asset['id']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    if payload.get("schema") != "flagquantum_example_assets_v1":
        raise ValueError("unsupported example asset manifest")
    for asset in payload["assets"]:
        destination = args.output / asset["path"]
        if not args.verify_only and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(asset["url"], destination)
        if not destination.exists():
            raise FileNotFoundError(destination)
        verify_asset(destination, asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
