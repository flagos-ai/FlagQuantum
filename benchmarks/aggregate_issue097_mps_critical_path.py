"""Audit and aggregate ISSUE-097 raw critical-path reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/manifests/issue097_mps_critical_path_v1.json"
_TRACE_NAME = re.compile(r'^\s*"name":\s*("(?:[^"\\]|\\.)*")')
_TRACE_DURATION = re.compile(r'^\s*"dur":\s*([0-9.eE+-]+)')


def _trace_summary(path: Path) -> dict:
    """Stream a Chrome trace so multi-GB evidence is validated in bounded memory."""
    labels: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_seconds": 0.0}
    )
    user_annotation = False
    name = None
    nccl_event_count = 0
    nccl_send_recv_count = 0
    with path.open(errors="replace") as stream:
        for line in stream:
            if '"cat": "user_annotation"' in line:
                user_annotation = True
                name = None
                continue
            match = _TRACE_NAME.match(line)
            if match:
                parsed = json.loads(match.group(1))
                if parsed.startswith("nccl"):
                    nccl_event_count += 1
                    if "SendRecv" in parsed or parsed in {"nccl:send", "nccl:recv"}:
                        nccl_send_recv_count += 1
                if user_annotation:
                    name = parsed
                continue
            match = _TRACE_DURATION.match(line)
            if user_annotation and name is not None and match:
                labels[name]["count"] += 1
                labels[name]["total_seconds"] += float(match.group(1)) / 1_000_000.0
                user_annotation = False
                name = None
    return {
        "labels": dict(labels),
        "nccl_event_count": nccl_event_count,
        "nccl_send_recv_count": nccl_send_recv_count,
    }


def _expected_workload(manifest: dict, case: str, world: int) -> dict:
    case_payload = manifest["cases"][case]
    return {
        "manifest_schema": manifest["schema"],
        "case": case,
        "world_size": world,
        "n_wires": max(world, int(case_payload["sites_per_rank"]) * world),
        **case_payload,
        "steps": int(manifest["warmup_steps"]) + int(manifest["retained_warm_steps"]),
        "dtype": manifest["dtype"],
        "optimizer": manifest["optimizer"],
    }


def _stable_environment(environment: dict) -> dict:
    return {
        key: value
        for key, value in environment.items()
        if key not in {
            "world_size", "local_world_size", "node_count", "rank_placement",
            "source_tree_dirty", "source_diff_sha256", "source_status_sha256",
        }
    }


def aggregate(paths: list[Path], *, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text())
    reports = [json.loads(path.read_text()) for path in paths]
    expected = {
        (case, int(world))
        for case in manifest["cases"]
        for world in manifest["world_sizes"]
    }
    observed = {(item["workload_manifest"]["case"], item["world_size"]) for item in reports}
    blockers = []
    if observed != expected or len(reports) != len(expected):
        blockers.append("incomplete_case_world_size_matrix")
    if any(
        item["workload_manifest"]
        != _expected_workload(
            manifest, item["workload_manifest"]["case"], int(item["world_size"])
        )
        for item in reports
    ):
        blockers.append("workload_manifest_mismatch")
    if any(not item["phase_reconciliation_passed"] for item in reports):
        blockers.append("phase_reconciliation_failed")
    retained = int(manifest["retained_warm_steps"])
    warmup = int(manifest["warmup_steps"])
    if any(item["warm_sample_count_across_ranks"] != retained * item["world_size"] for item in reports):
        blockers.append("twenty_warm_samples_per_rank_missing")
    if any(
        item.get("warmup_steps") != warmup
        or {
            (sample["rank"], sample["step"])
            for sample in item["rank_step_samples"]
        }
        != {
            (rank, step)
            for rank in range(int(item["world_size"]))
            for step in range(warmup + retained)
        }
        for item in reports
    ):
        blockers.append("rank_step_sample_matrix_invalid")
    environments = {
        json.dumps(_stable_environment(item["environment_manifest"]), sort_keys=True)
        for item in reports
    }
    if len(environments) != 1:
        blockers.append("source_or_environment_snapshot_mismatch")
    for item in reports:
        environment = item["environment_manifest"]
        world = int(item["world_size"])
        placements = environment.get("rank_placement", ())
        setup = item.get("cold_setup_seconds_by_rank", {})
        if (
            environment.get("backend") != "nccl"
            or int(environment.get("world_size", -1)) != world
            or int(environment.get("local_world_size", 0)) <= 0
            or int(environment.get("node_count", 0)) <= 0
            or len(placements) != world
            or {int(value["rank"]) for value in placements} != set(range(world))
        ):
            blockers.append("topology_environment_manifest_invalid")
            break
        if set(setup) != {str(rank) for rank in range(world)} or any(
            not math.isfinite(float(values.get(name, -1.0)))
            or float(values.get(name, -1.0)) < 0.0
            for values in setup.values()
            for name in ("process_group_seconds", "communication_seconds")
        ):
            blockers.append("cold_setup_evidence_invalid")
            break
    points = []
    for report in reports:
        phases = defaultdict(list)
        end_to_end = []
        for sample in report["rank_step_samples"]:
            if sample["sample_class"] != "warm":
                continue
            end_to_end.append(sample["end_to_end_seconds"])
            for name, seconds in sample["phase_seconds"].items():
                phases[name].append(seconds)
        medians = {name: statistics.median(values) for name, values in phases.items()}
        dominant = max((name for name in medians if name != "unattributed"), key=medians.get)
        trace_paths = tuple(Path(value) for value in report.get("trace_paths", ()))
        if len(trace_paths) != report["world_size"] or len(set(trace_paths)) != len(trace_paths):
            blockers.append("rank_trace_set_invalid")
            trace_paths = ()
        trace_totals: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"count": 0, "total_seconds": 0.0}
        )
        required_operations = tuple(
            f"flagquantum::mps::{name}"
            for name in manifest["required_phase_labels"]
            if report["world_size"] > 1 or name not in {"two_site_split", "qr", "svd"}
        )
        required_communication = tuple(
            f"flagquantum::mps::{name}"
            for name in manifest["required_communication_labels"]
            if report["world_size"] > 1 or name not in {"p2p_send", "p2p_recv"}
        )
        for trace_path in trace_paths:
            resolved = trace_path if trace_path.is_absolute() else REPO_ROOT / trace_path
            if not resolved.is_file():
                blockers.append("rank_trace_missing")
                continue
            summary = _trace_summary(resolved)
            if summary["nccl_event_count"] == 0 or (
                report["world_size"] > 1 and summary["nccl_send_recv_count"] == 0
            ):
                blockers.append("nccl_trace_evidence_missing")
            for label, values in summary["labels"].items():
                if label.startswith("flagquantum::mps::"):
                    trace_totals[label]["count"] += int(values["count"])
                    trace_totals[label]["total_seconds"] += float(values["total_seconds"])
        missing = [
            label
            for label in required_operations + required_communication
            if trace_totals.get(label, {}).get("count", 0) == 0
        ]
        if missing:
            blockers.append("required_trace_labels_missing")
        points.append({
            "case": report["workload_manifest"]["case"],
            "world_size": report["world_size"],
            "median_end_to_end_seconds": statistics.median(end_to_end),
            "median_phase_seconds": medians,
            "dominant_measured_phase": dominant,
            "message_totals_by_payload_class": report["message_totals_by_payload_class"],
            "maximum_reconciliation_relative_error": report["maximum_reconciliation_relative_error"],
            "trace_operation_totals": dict(trace_totals),
            "cold_setup_seconds_by_rank": report.get("cold_setup_seconds_by_rank", {}),
        })
    blockers = sorted(set(blockers))
    return {
        "schema": "flagquantum.issue097.mps_critical_path_baseline.v1",
        "matrix_complete": observed == expected,
        "reports": len(reports),
        "rank_traces": sum(item["world_size"] for item in reports),
        "warm_samples": sum(item["warm_sample_count_across_ranks"] for item in reports),
        "bottleneck_points": sorted(points, key=lambda item: (item["case"], item["world_size"])),
        "blockers": blockers,
        "baseline_gate_passed": not blockers,
        "performance_claim_allowed": False,
        "scalability_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    paths = sorted(path for path in args.input.glob("*.json") if ".trace." not in path.name)
    payload = aggregate(paths, manifest_path=args.manifest)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["baseline_gate_passed"]:
        raise SystemExit("ISSUE-097 baseline audit failed: " + ",".join(payload["blockers"]))


if __name__ == "__main__":
    main()
