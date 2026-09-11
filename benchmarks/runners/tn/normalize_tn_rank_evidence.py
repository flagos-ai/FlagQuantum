#!/usr/bin/env python3
"""Normalize existing TN rank evidence without changing measured values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tn_sliced_reverse_nccl import _audit_evidence_fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            _audit_evidence_fields(
                payload["rank_results"], world_size=int(payload["world_size"])
            )
        )
        payload["report_schema_normalized_from_rank_results"] = True
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
