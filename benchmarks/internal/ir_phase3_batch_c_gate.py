"""Enforce the approved private Phase 3 Batch C performance budget."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from benchmarks.internal.ir_phase3_batch_c_baseline import measure_case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase3_batch_c_performance_budget_candidate.json"
)


def evaluate(budget_path: Path, *, iterations: int, warmup: int) -> dict[str, Any]:
    budget_path = budget_path.resolve()
    payload = json.loads(budget_path.read_text(encoding="utf-8"))
    if payload["approval"]["approval_command"] != (
        "approve IR-PHASE3-BATCH-C-PERFORMANCE-BUDGET"
    ):
        raise ValueError("unexpected Batch C budget approval command")
    budgets = {int(item["operation_count"]): item for item in payload["budgets"]}
    if set(budgets) != {10, 100, 1000, 10000}:
        raise ValueError("Batch C budget must cover 10/100/1K/10K operations")
    cases = []
    for operation_count in (10, 100, 1000, 10000):
        case = measure_case(
            operation_count,
            iterations=iterations,
            warmup=warmup,
        )
        budget = budgets[operation_count]
        case["budget"] = budget
        case["latency_passed"] = (
            case["seal_and_verify_p95_ms"] <= budget["seal_and_verify_p95_ms_max"]
        )
        case["memory_passed"] = (
            case["seal_and_verify_peak_host_memory_bytes"]
            <= budget["seal_and_verify_peak_host_memory_bytes_max"]
        )
        case["passed"] = bool(
            case["latency_passed"]
            and case["memory_passed"]
            and case["deterministic_identity"]
        )
        cases.append(case)
    passed = all(case["passed"] for case in cases)
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "claim_scope": "Phase 3 Batch C private CPU budget; not a public SLA",
        "budget_path": str(budget_path.relative_to(ROOT)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "pipeline": "canonical payload validation, sealing, identity, and verification",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    args = parser.parse_args()
    if args.iterations < 5 or args.warmup < 1:
        raise ValueError("iterations must be >= 5 and warmup must be >= 1")
    result = evaluate(args.budget, iterations=args.iterations, warmup=args.warmup)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
