"""Bind matched OOM, completion, crash, and resume capacity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    baseline: dict[str, Any],
    uninterrupted: dict[str, Any],
    crash: dict[str, Any],
    resume: dict[str, Any],
    recovered: dict[str, Any],
) -> dict[str, Any]:
    hashes = {
        baseline.get("workload_sha256"),
        uninterrupted.get("workload_sha256"),
        recovered.get("workload_sha256"),
    }
    if len(hashes) != 1 or None in hashes:
        raise ValueError("capacity recovery workloads do not match")
    baseline_ranks = baseline.get("ranks", [])
    if len(baseline_ranks) != 1 or not (
        baseline_ranks[0].get("status") == "expected_oom"
        and baseline_ranks[0].get("single_device_oom_observed") is True
        and baseline_ranks[0].get("expectation_met") is True
    ):
        raise ValueError("baseline does not prove measured single-device OOM")
    if not (
        crash.get("status") == "failed_closed"
        and crash.get("reason") == "peer_launcher_failure"
        and crash.get("cleanup_verified") is True
        and crash.get("all_launchers_exited") is True
    ):
        raise ValueError("crash watchdog did not prove bounded cleanup")
    if not (
        resume.get("status") == "passed"
        and resume.get("reason") == "completed"
        and resume.get("cleanup_verified") is True
        and resume.get("all_launchers_exited") is True
    ):
        raise ValueError("resume watchdog did not prove completion")
    reference_ranks = uninterrupted.get("ranks", [])
    recovered_ranks = recovered.get("ranks", [])
    if len(reference_ranks) < 2 or len(recovered_ranks) != len(reference_ranks):
        raise ValueError("capacity completion requires matching distributed ranks")
    reference_parameters = reference_ranks[0]["parameters"]
    reference_final_loss = reference_ranks[0]["training"]["losses"][-1]
    expected_ranks = set(range(len(recovered_ranks)))
    if {rank.get("rank") for rank in recovered_ranks} != expected_ranks:
        raise ValueError("recovered rank coverage is incomplete")
    hostnames = {rank.get("hostname") for rank in recovered_ranks}
    if None in hostnames or len(hostnames) < 2:
        raise ValueError("recovered capacity run is not multi-node")
    for rank in recovered_ranks:
        training = rank.get("training", {})
        if not (
            rank.get("status") == "passed"
            and rank.get("resume_requested") is True
            and rank.get("checkpoint_enabled") is True
            and rank.get("parameters") == reference_parameters
            and training.get("start_step") == 1
            and training.get("completed_steps") == 2
            and training.get("losses") == [reference_final_loss]
            and training.get("distribution_semantics") == "sharded_across_ranks"
        ):
            raise ValueError("recovered rank differs from uninterrupted capacity run")
    return {
        "schema": "flagquantum.statevector.capacity_recovery.v1",
        "artifact_classification": "measured_development_capacity_recovery",
        "release_evidence": False,
        "recovery_claim_allowed": False,
        "development_capacity_recovery_observed": True,
        "workload_sha256": next(iter(hashes)),
        "n_wires": recovered_ranks[0]["workload"]["n_wires"],
        "world_size": len(recovered_ranks),
        "node_count": len(hostnames),
        "checkpoint_generation": 1,
        "completed_steps": 2,
        "final_loss": reference_final_loss,
        "final_parameters": reference_parameters,
        "peak_allocated_bytes_by_rank": [
            rank["device"]["peak_allocated_bytes"] for rank in recovered_ranks
        ],
        "crash_detection_seconds": crash.get("failure_detected_seconds"),
        "crash_cleanup_seconds": crash.get("cleanup_elapsed_seconds"),
        "resume_elapsed_seconds": resume.get("elapsed_seconds"),
        "blockers": [
            "development_capacity_threshold_not_preregistered",
            "repeated_capacity_fault_soak_not_measured",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--crash-watchdog", type=Path, required=True)
    parser.add_argument("--resume-watchdog", type=Path, required=True)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        _load(args.baseline),
        _load(args.uninterrupted),
        _load(args.crash_watchdog),
        _load(args.resume_watchdog),
        _load(args.recovered),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
