"""Portable runner smoke test and environment manifest producer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:  # Supports both ``python -m`` and direct script execution.
    from .contract import runtime_metadata, validate_payload, write_json_atomic
except ImportError:  # pragma: no cover - direct CLI path
    from contract import runtime_metadata, validate_payload, write_json_atomic


def build_payload() -> dict[str, object]:
    return runtime_metadata(
        runner="environment_probe",
        schema="flagquantum.benchmark.environment.v1",
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        device=os.environ.get("CUDA_VISIBLE_DEVICES", "unrestricted"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = build_payload()
    validate_payload(payload)
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
