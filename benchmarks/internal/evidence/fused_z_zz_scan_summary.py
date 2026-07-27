"""Fail-closed aggregation for ISSUE-098 fused scan evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(paths: list[Path]) -> dict:
    records = [json.loads(path.read_text()) for path in paths]
    expected = {1, 2, 4, 8}
    blockers = []
    if {item.get("world_size") for item in records} != expected:
        blockers.append("incomplete_world_size_matrix")
    if any(item.get("backend") != "nccl" for item in records):
        blockers.append("nccl_evidence_missing")
    if any(item.get("objective_scan_pairs") != 1 for item in records):
        blockers.append("more_than_one_scan_pair")
    if any(item.get("environment_transfer_launch_reduction", 0.0) < 0.35 for item in records):
        blockers.append("launch_reduction_below_35_percent")
    if any(not item.get("passed", False) for item in records):
        blockers.append("numerical_or_scan_gate_failed")
    return {
        "schema": "flagquantum.issue098.fused_z_zz_scan_matrix.v1",
        "world_sizes": sorted(item["world_size"] for item in records),
        "device": records[0].get("device") if records else None,
        "minimum_launch_reduction": min(
            (item["environment_transfer_launch_reduction"] for item in records),
            default=0.0,
        ),
        "maximum_value_error": max((item["max_value_error"] for item in records), default=float("inf")),
        "maximum_gradient_error": max((item["max_gradient_error"] for item in records), default=float("inf")),
        "scan_pairs_per_probe_microbatch": 1 if records else None,
        "records": [str(path) for path in paths],
        "blockers": blockers,
        "passed": not blockers,
        "performance_claim_allowed": False,
        "scalability_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*gpu.json"))
    payload = aggregate(paths)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["passed"]:
        raise SystemExit("ISSUE-098 evidence failed: " + ",".join(payload["blockers"]))


if __name__ == "__main__":
    main()
