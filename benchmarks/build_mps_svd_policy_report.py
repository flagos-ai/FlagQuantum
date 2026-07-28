"""Build auditable MPS SVD-driver policy evidence."""

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
    selections = []
    for bond in raw["bonds"]:
        records = [
            record
            for record in raw["records"]
            if record["dimension"] == 2 * bond and record["supported"]
        ]
        exact = min(
            (
                record
                for record in records
                if record["relative_reconstruction_residual"] <= 1e-5
                and record["driver"] != "gesvda"
            ),
            key=lambda record: record["mean_seconds"],
        )
        approximate = min(
            (
                record
                for record in records
                if record["relative_reconstruction_residual"] <= 1e-4
            ),
            key=lambda record: record["mean_seconds"],
        )
        selections.append(
            {
                "bond": bond,
                "dimension": 2 * bond,
                "exact_driver": exact["driver"],
                "exact_mean_seconds": exact["mean_seconds"],
                "exact_residual": exact["relative_reconstruction_residual"],
                "approximate_driver": approximate["driver"],
                "approximate_mean_seconds": approximate["mean_seconds"],
                "approximate_residual": approximate[
                    "relative_reconstruction_residual"
                ],
                "approximate_speedup": (
                    exact["mean_seconds"] / approximate["mean_seconds"]
                ),
            }
        )
    peak = max(
        record.get("peak_allocated_memory_bytes", 0)
        for record in raw["records"]
    )
    payload = {
        "schema": "flagquantum.mps_svd_policy_report.v1",
        "benchmark": "mps_svd_driver_calibration",
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
        "exact_residual_budget": 1e-5,
        "approximate_residual_budget": 1e-4,
        "records": raw["records"],
        "selections": selections,
        "passed": (
            raw["source_tree_dirty"] is False
            and len(selections) == len(raw["bonds"])
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "single_device_driver_calibration",
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
