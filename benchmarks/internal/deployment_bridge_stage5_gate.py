"""Enforce the approved private Deployment Bridge Stage 5 budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from benchmarks.internal.deployment_bridge_stage5_baseline import measure_case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "deployment_bridge_stage5_performance_budget_candidate.json"
)
DEFAULT_AUTHORIZATION = (
    ROOT / "contracts/deployment-bridge-stage5-performance-budget-authorization.json"
)
APPROVAL_COMMAND = "approve DEPLOYMENT-BRIDGE-STAGE5-PERFORMANCE-BUDGET"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    budget_path: Path,
    authorization_path: Path,
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    budget_path = budget_path.resolve()
    authorization_path = authorization_path.resolve()
    budget_payload = json.loads(budget_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "approved":
        raise ValueError("Stage 5 performance budget is not approved")
    if authorization.get("approval_command") != APPROVAL_COMMAND:
        raise ValueError("unexpected Stage 5 budget authorization command")
    approved_budget = authorization.get("approved_budget", {})
    if approved_budget.get("sha256") != _sha256(budget_path):
        raise ValueError("Stage 5 budget does not match its authorization")
    if budget_payload["approval"]["approval_command"] != APPROVAL_COMMAND:
        raise ValueError("unexpected Stage 5 budget approval command")
    budgets = {
        int(item["observation_count"]): item for item in budget_payload["budgets"]
    }
    if set(budgets) != {10, 100, 1000, 10000}:
        raise ValueError("Stage 5 budget must cover 10/100/1K/10K observations")
    cases = []
    for observation_count in (10, 100, 1000, 10000):
        case = measure_case(
            observation_count,
            iterations=iterations,
            warmup=warmup,
        )
        budget = budgets[observation_count]
        case["budget"] = budget
        case["latency_passed"] = (
            case["offline_observation_p95_ms"]
            <= budget["offline_observation_p95_ms_max"]
        )
        case["memory_passed"] = (
            case["offline_observation_peak_host_memory_bytes"]
            <= budget["offline_observation_peak_host_memory_bytes_max"]
        )
        case["passed"] = bool(
            case["latency_passed"]
            and case["memory_passed"]
            and case["deterministic_evidence_identity"]
        )
        cases.append(case)
    passed = all(case["passed"] for case in cases)
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "claim_scope": "Deployment Bridge Stage 5 private CPU budget; not a public SLA",
        "authorization_path": str(authorization_path.relative_to(ROOT)),
        "budget_path": str(budget_path.relative_to(ROOT)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "pipeline": "anonymous offline accepted sandbox observation",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    args = parser.parse_args()
    if args.iterations < 5 or args.warmup < 1:
        raise ValueError("iterations must be >= 5 and warmup must be >= 1")
    result = evaluate(
        args.budget,
        args.authorization,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
