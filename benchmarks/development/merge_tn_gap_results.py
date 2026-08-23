"""Merge native and isolated results into a fail-closed TN gap report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.development.tn_gap_common import validate_backend_result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagquantum", type=Path, required=True)
    parser.add_argument("--cotengra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def merge(flagquantum: dict[str, Any], cotengra: dict[str, Any]) -> dict[str, Any]:
    validate_backend_result(flagquantum, expected_backend="flagquantum")
    validate_backend_result(cotengra, expected_backend="cotengra")
    if flagquantum["workload_identity"] != cotengra["workload_identity"]:
        raise ValueError("cannot merge TN gap results for different workloads")
    comparable = (
        flagquantum["status"] == "completed"
        and cotengra["status"] == "completed"
        and flagquantum["search_budget_compliant"]
        and cotengra["search_budget_compliant"]
        and flagquantum.get("search_budget_seconds")
        == cotengra.get("search_budget_seconds")
    )
    ratios: dict[str, float | None] = {}
    largest_gap = "comparison unavailable"
    next_target = "establish a completed, budget-compliant comparison"
    if comparable:
        fq_metrics = flagquantum["metrics"]
        ctg_metrics = cotengra["metrics"]
        fq_flops = fq_metrics.get(
            "common_model_estimated_flops",
            fq_metrics["estimated_flops"],
        )
        ctg_flops = ctg_metrics.get(
            "common_model_estimated_flops",
            ctg_metrics["estimated_flops"],
        )
        fq_peak = fq_metrics.get(
            "common_model_largest_intermediate_elements",
            fq_metrics["largest_intermediate_elements"],
        )
        ctg_peak = ctg_metrics.get(
            "common_model_largest_intermediate_elements",
            ctg_metrics["largest_intermediate_elements"],
        )
        ratios = {
            "estimated_flops_flagquantum_over_cotengra": _ratio(
                fq_flops, ctg_flops
            ),
            "largest_intermediate_flagquantum_over_cotengra": _ratio(
                fq_peak,
                ctg_peak,
            ),
            "search_time_flagquantum_over_cotengra": _ratio(
                fq_metrics["search_time_seconds"],
                ctg_metrics["search_time_seconds"],
            ),
        }
        candidates = {
            "path_flops": ratios["estimated_flops_flagquantum_over_cotengra"],
            "largest_intermediate": ratios[
                "largest_intermediate_flagquantum_over_cotengra"
            ],
            "search_time": ratios["search_time_flagquantum_over_cotengra"],
        }
        largest_gap = max(
            candidates,
            key=lambda name: (
                float("-inf")
                if candidates[name] is None
                else float(candidates[name])
            ),
        )
        next_target = {
            "path_flops": "improve native multi-start path search",
            "largest_intermediate": "improve memory-aware path search and slicing",
            "search_time": "reduce native planning overhead or bypass reconfiguration",
        }[largest_gap]
    return {
        "schema_version": 1,
        "workload_identity": flagquantum["workload_identity"],
        "workload_name": flagquantum.get("workload_name"),
        "comparison_scope": "planning",
        "search_budget_seconds": {
            "flagquantum": flagquantum.get("search_budget_seconds"),
            "cotengra": cotengra.get("search_budget_seconds"),
        },
        "flagquantum": flagquantum,
        "cotengra": cotengra,
        "comparable": comparable,
        "ratios": ratios,
        "correctness": {"topology_identity_equal": True},
        "largest_gap": largest_gap,
        "next_optimization_target": next_target,
        "claim_boundary": (
            "Planning-only development evidence; not execution, gradient, "
            "production, or scalability evidence."
        ),
    }


def main() -> None:
    arguments = _arguments()
    flagquantum = json.loads(arguments.flagquantum.read_text(encoding="utf-8"))
    cotengra = json.loads(arguments.cotengra.read_text(encoding="utf-8"))
    payload = merge(flagquantum, cotengra)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
