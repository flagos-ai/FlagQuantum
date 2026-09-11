#!/usr/bin/env python3
"""Build the fail-closed external-comparison and ablation report for Gate 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    *,
    external_path: Path,
    operator_off_path: Path,
    operator_on_path: Path,
    communication_off_path: Path,
    communication_on_path: Path,
) -> dict[str, Any]:
    external = _load(external_path)
    operator_off, operator_on = _load(operator_off_path), _load(operator_on_path)
    communication_off = _load(communication_off_path)
    communication_on = _load(communication_on_path)
    blockers: list[str] = []

    isolation = external.get("external_baseline_isolation", {})
    external_passed = bool(
        external.get("correctness", {}).get("passed")
        and external.get("stability", {}).get("comparison_claim_allowed")
        and external.get("world_size") == 1
        and isolation.get("purpose") == "benchmark_only"
        and isolation.get("flagquantum_runtime_dependency") is False
    )
    if not external_passed:
        blockers.append("external_comparison_failed")

    same_operator_workload = operator_off.get("workload") == operator_on.get("workload")
    operator_correct = all(
        item.get("correctness", {}).get("reference_gradient_absolute_error_max", 1.0)
        <= 3e-5
        for item in (operator_off, operator_on)
    )
    operator_speedup = (
        operator_off["backward"]["median_seconds"]
        / operator_on["backward"]["median_seconds"]
    )
    operator_passed = bool(
        same_operator_workload
        and operator_correct
        and operator_off.get("benchmark_protocol", {}).get("triton_vjp_adjoint") is False
        and operator_on.get("benchmark_protocol", {}).get("triton_vjp_adjoint") is True
        and operator_speedup > 1.0
    )
    if not operator_passed:
        blockers.append("operator_ablation_failed")

    same_communication_workload = (
        communication_off.get("workload") == communication_on.get("workload")
    )
    communication_correct = all(
        item.get("correctness", {}).get("reference_gradient_absolute_error_max", 1.0)
        <= 3e-5
        for item in (communication_off, communication_on)
    )
    bytes_off = communication_off["backward_communication_bytes_per_rank_max"]
    bytes_on = communication_on["backward_communication_bytes_per_rank_max"]
    communication_passed = bool(
        same_communication_workload
        and communication_correct
        and communication_off.get("benchmark_protocol", {}).get("cross_shard_cx_pack")
        is False
        and communication_on.get("benchmark_protocol", {}).get("cross_shard_cx_pack")
        is True
        and bytes_on < bytes_off
    )
    if not communication_passed:
        blockers.append("communication_ablation_failed")

    return {
        "schema": "flagquantum.statevector_gate2_report.v1",
        "gate": "external_system_comparison_and_communication_operator_ablations",
        "artifact_class": "measured_development_evidence_report",
        "passed": not blockers,
        "blockers": blockers,
        "release_gate_allowed": False,
        "external_comparison": {
            "passed": external_passed,
            "artifact": str(external_path),
            "provider": isolation.get("provider"),
            "flagquantum_median_seconds": external["flagquantum"]["median_seconds"],
            "external_median_seconds": external["custatevec"]["median_seconds"],
            "external_over_flagquantum_speedup": external[
                "speedup_custatevec_over_flagquantum"
            ],
            "max_abs_error": external["correctness"]["max_abs_error"],
            "isolated_benchmark_environment": isolation,
        },
        "operator_ablation": {
            "passed": operator_passed,
            "off_artifact": str(operator_off_path),
            "on_artifact": str(operator_on_path),
            "backward_speedup": operator_speedup,
            "end_to_end_speedup": (
                operator_off["end_to_end"]["median_seconds"]
                / operator_on["end_to_end"]["median_seconds"]
            ),
        },
        "communication_ablation": {
            "passed": communication_passed,
            "off_artifact": str(communication_off_path),
            "on_artifact": str(communication_on_path),
            "backward_bytes_reduction_fraction": (bytes_off - bytes_on) / bytes_off,
            "end_to_end_speedup": (
                communication_off["end_to_end"]["median_seconds"]
                / communication_on["end_to_end"]["median_seconds"]
            ),
        },
        "claim_scope": (
            "Gate 2 development evidence only; external cuStateVec is single-device "
            "forward-only and is never a FlagQuantum runtime dependency."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--operator-off", type=Path, required=True)
    parser.add_argument("--operator-on", type=Path, required=True)
    parser.add_argument("--communication-off", type=Path, required=True)
    parser.add_argument("--communication-on", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        external_path=args.external,
        operator_off_path=args.operator_off,
        operator_on_path=args.operator_on,
        communication_off_path=args.communication_off,
        communication_on_path=args.communication_on,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
