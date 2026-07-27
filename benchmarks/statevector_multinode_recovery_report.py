"""Validate and aggregate a statevector crash/checkpoint/resume campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_json(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON rank record in {path}")


def build_report(
    crash: dict[str, Any],
    resume: dict[str, Any],
    rank_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not (
        crash.get("schema") == "flagquantum.multinode_watchdog.v1"
        and crash.get("status") == "failed_closed"
        and crash.get("reason") == "peer_launcher_failure"
        and crash.get("cleanup_verified") is True
        and crash.get("all_launchers_exited") is True
        and not crash.get("cleanup_failures")
    ):
        raise ValueError("crash watchdog did not prove fail-closed cleanup")
    if not (
        resume.get("schema") == "flagquantum.multinode_watchdog.v1"
        and resume.get("status") == "passed"
        and resume.get("reason") == "completed"
        and resume.get("cleanup_verified") is True
        and resume.get("all_launchers_exited") is True
    ):
        raise ValueError("resume watchdog did not prove successful completion")
    if len(rank_records) < 2:
        raise ValueError("at least two recovered rank records are required")
    ranks = set()
    for record in rank_records:
        summary = record.get("summary", {})
        rank = summary.get("rank")
        ranks.add(rank)
        if not (
            record.get("cleanup_verified") is True
            and record.get("resumed_parameters") == record.get("reference_parameters")
            and summary.get("distribution_semantics") == "sharded_across_ranks"
            and summary.get("start_step", 0) > 0
            and summary.get("completed_steps", 0) > summary.get("start_step", 0)
            and summary.get("node_count", 0) >= 2
            and summary.get("world_size") == len(rank_records)
            and summary.get("losses") == record.get("reference_tail_losses")
        ):
            raise ValueError("rank record does not prove exact resumed training")
    if ranks != set(range(len(rank_records))):
        raise ValueError("recovered rank coverage is incomplete")
    parameters = [record["resumed_parameters"] for record in rank_records]
    losses = [record["summary"]["losses"] for record in rank_records]
    if any(value != parameters[0] for value in parameters[1:]):
        raise ValueError("recovered ranks disagree on parameters")
    if any(value != losses[0] for value in losses[1:]):
        raise ValueError("recovered ranks disagree on losses")
    return {
        "schema": "flagquantum.statevector.multinode_recovery.v1",
        "artifact_classification": "measured_development_recovery",
        "release_evidence": False,
        "recovery_claim_allowed": False,
        "development_multinode_recovery_observed": True,
        "world_size": len(rank_records),
        "node_count": rank_records[0]["summary"]["node_count"],
        "checkpoint_generation": rank_records[0]["summary"]["start_step"],
        "completed_steps": rank_records[0]["summary"]["completed_steps"],
        "losses": losses[0],
        "parameters": parameters[0],
        "crash_detection_seconds": crash.get("failure_detected_seconds"),
        "crash_cleanup_seconds": crash.get("cleanup_elapsed_seconds"),
        "resume_elapsed_seconds": resume.get("elapsed_seconds"),
        "rank_optimizer_ownership": [
            record["summary"]["optimizer_ownership"] for record in rank_records
        ],
        "blockers": [
            "development_recovery_campaign_not_preregistered",
            "repeated_fault_and_checkpoint_corruption_matrix_not_measured",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash-watchdog", type=Path, required=True)
    parser.add_argument("--resume-watchdog", type=Path, required=True)
    parser.add_argument("--rank-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        _load(args.crash_watchdog),
        _load(args.resume_watchdog),
        [_last_json(path) for path in args.rank_log],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
