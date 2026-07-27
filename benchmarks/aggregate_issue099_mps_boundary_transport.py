"""Aggregate ISSUE-099 measured 2/8-GPU boundary transport evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for path in args.input.glob("*gpu.json"))
    records = [json.loads(path.read_text()) for path in paths]
    blockers = []
    if {item.get("world_size") for item in records} != {2, 8}:
        blockers.append("two_and_eight_gpu_matrix_missing")
    if any(not item.get("passed") for item in records):
        blockers.append("transport_gate_failed")
    if any(item.get("message_count_reduction", 0.0) < 0.40 for item in records):
        blockers.append("message_reduction_below_40_percent")
    if any(item.get("logical_byte_growth", 1.0) > 0.05 for item in records):
        blockers.append("logical_byte_growth_above_5_percent")
    if any(item.get("p2p_wait_reduction", 0.0) <= 0.0 for item in records):
        blockers.append("p2p_wait_not_reduced")
    if any(not item.get("shape_mismatch_fault", {}).get("bounded_cleanup") for item in records):
        blockers.append("fault_cleanup_missing")
    payload = {
        "schema": "flagquantum.issue099.mps_boundary_transport_matrix.v1",
        "world_sizes": sorted(item["world_size"] for item in records),
        "minimum_message_count_reduction": min((item["message_count_reduction"] for item in records), default=0.0),
        "maximum_logical_byte_growth": max((item["logical_byte_growth"] for item in records), default=1.0),
        "minimum_p2p_wait_reduction": min((item["p2p_wait_reduction"] for item in records), default=0.0),
        "minimum_overlap_work_seconds": min((item["overlap_work_seconds"] for item in records), default=0.0),
        "trace_count": sum(len(item.get("trace_paths", ())) for item in records),
        "records": [str(path) for path in paths],
        "timeout_contract_test": "tests/unit/test_issue094_mps_p2p_diagnostics.py",
        "packed_fault_test": "tests/distributed/test_issue099_mps_packed_transport.py",
        "blockers": blockers,
        "passed": not blockers,
        "performance_claim_allowed": False,
        "scalability_claim_allowed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if blockers:
        raise SystemExit("ISSUE-099 evidence failed: " + ",".join(blockers))


if __name__ == "__main__":
    main()
