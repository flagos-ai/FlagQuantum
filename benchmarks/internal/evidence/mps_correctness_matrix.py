"""Aggregate ISSUE-091 per-topology correctness cases into the audit matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flagquantum.testing import require_mps_numerical_certification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(path.read_text()) for path in args.inputs]
    payload = {
        "schema": "flagquantum.issue091.mps_correctness_matrix.v1",
        "dtypes": sorted({case["dtype"] for case in cases}),
        "steps": min(int(case["steps"]) for case in cases),
        "seed": 91_052,
        "tolerances": {
            "value_atol": 5e-5,
            "gradient_atol": 3e-4,
            "parameter_atol": 3e-5,
            "directional_atol": 5e-4,
            "approximate_value_atol": 1e-4,
            "approximate_gradient_atol": 8e-4,
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "source_artifacts": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.inputs
        ],
    }
    require_mps_numerical_certification(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": True}))


if __name__ == "__main__":
    main()
