"""Contract-aware adapter for statevector training scaling reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from benchmarks.statevector_training_scaling_report import build_report
except ModuleNotFoundError:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.statevector_training_scaling_report import build_report

try:
    from .contract import runtime_metadata, write_json_atomic
except ImportError:  # direct execution
    from contract import runtime_metadata, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report([json.loads(p.read_text(encoding="utf-8")) for p in args.artifacts])
    payload = runtime_metadata(
        runner="statevector_training_scaling",
        schema="flagquantum.benchmark.statevector_training_scaling.v1",
        report=report,
    )
    write_json_atomic(args.json_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
