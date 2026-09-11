"""Build auditable no-truncation MPS QR calibration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    peak = max(
        record["full_mps_update"]["peak_allocated_memory_bytes"]
        for record in raw["records"]
    )
    records = []
    for record in raw["records"]:
        direct = record["direct_qr"]["mean_seconds"]
        full = record["full_mps_update"]["mean_seconds"]
        records.append(
            {
                **record,
                "transfer_overhead_fraction": full / direct - 1,
                "correctness_passed": (
                    record["relative_reconstruction_residual"] <= 2e-6
                    and record["normalized_orthogonality_error"] <= 2e-6
                ),
            }
        )
    payload = {
        "schema": "flagquantum.mps_qr_policy_report.v1",
        "benchmark": "mps_no_truncation_qr_calibration",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "distribution_semantics": "single_device_fast_path",
        "world_size": 1,
        "local_world_size": 1,
        "node_count": 1,
        "rank_placement": [
            {
                "rank": 0,
                "local_rank": 0,
                "hostname": "p-kt-lc-a800-04",
                "device": "cuda:0",
            }
        ],
        "local_memory_bytes_by_rank": [peak],
        "communication_bytes": 0,
        "device": raw["device"],
        "dtype": raw["dtype"],
        "source_commit": raw["source_commit"],
        "source_tree_dirty": raw["source_tree_dirty"],
        "warmups": raw["warmups"],
        "samples": raw["samples"],
        "records": records,
        "selected_policy": {
            "mode": "reduced",
            "implementation": "torch.linalg.qr",
            "tiny_column_fast_path_max_columns": 2,
            "shape_scope": "batch1_complex64_2chi_by_chi",
        },
        "passed": (
            raw["source_tree_dirty"] is False
            and all(record["correctness_passed"] for record in records)
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "single_device_qr_calibration",
            "a800_only_portability_not_established",
            "single_run_unsigned_development_evidence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
