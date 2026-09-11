"""Run CoTenGra only inside an isolated comparison environment."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

from benchmarks.runners.tn.tn_gap_common import (
    RESULT_SCHEMA,
    load_workload,
    replay_pair_path_metrics,
    write_result,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--search-budget-seconds", type=float, default=1.0)
    parser.add_argument("--max-repeats", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-peak-elements", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _unavailable(arguments, workload, error: Exception) -> dict:
    return {
        "schema_version": RESULT_SCHEMA,
        "backend": "cotengra",
        "backend_version": "unavailable",
        "workload_identity": workload.identity,
        "workload_name": workload.name,
        "comparison_scope": "planning",
        "search_budget_seconds": arguments.search_budget_seconds,
        "status": "unavailable",
        "error_type": type(error).__name__,
        "error": str(error),
        "metrics": {},
        "environment": {"python": platform.python_version()},
        "distribution_semantics": "planning_only",
        "scalability_claim_allowed": False,
    }


def main() -> None:
    arguments = _arguments()
    workload = load_workload(arguments.workload)
    try:
        import cotengra as ctg
    except ImportError as error:
        payload = _unavailable(arguments, workload, error)
    else:
        random.seed(int(arguments.seed))
        slicing_opts = None
        if arguments.target_peak_elements is not None:
            slicing_opts = {"target_size": int(arguments.target_peak_elements)}
        optimizer = ctg.HyperOptimizer(
            methods=["greedy", "random-greedy"],
            minimize="combo",
            max_repeats=int(arguments.max_repeats),
            max_time=float(arguments.search_budget_seconds),
            slicing_opts=slicing_opts,
            parallel=False,
            progbar=False,
        )
        started = time.perf_counter()
        try:
            tree = optimizer.search(
                inputs=workload.inputs,
                output=workload.output,
                size_dict=workload.size_dict,
            )
        except Exception as error:
            elapsed = time.perf_counter() - started
            payload = {
                "schema_version": RESULT_SCHEMA,
                "backend": "cotengra",
                "backend_version": getattr(ctg, "__version__", "unknown"),
                "workload_identity": workload.identity,
                "workload_name": workload.name,
                "comparison_scope": "planning",
                "search_budget_seconds": arguments.search_budget_seconds,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "metrics": {"search_time_seconds": elapsed},
                "environment": {
                    "python": platform.python_version(),
                    "isolation": "temporary_target_directory",
                },
                "distribution_semantics": "planning_only",
                "scalability_claim_allowed": False,
            }
        else:
            elapsed = time.perf_counter() - started
            sliced = tuple(sorted(map(str, tree.sliced_inds)))
            common_metrics = replay_pair_path_metrics(
                workload,
                tree.get_path(),
            )
            budget_compliant = (
                arguments.search_budget_seconds <= 0
                or elapsed <= max(0.05, arguments.search_budget_seconds * 1.10)
            )
            payload = {
                "schema_version": RESULT_SCHEMA,
                "backend": "cotengra",
                "backend_version": getattr(ctg, "__version__", "unknown"),
                "workload_identity": workload.identity,
                "workload_name": workload.name,
                "comparison_scope": "planning",
                "search_budget_seconds": arguments.search_budget_seconds,
                "status": "completed",
                "search_budget_compliant": budget_compliant,
                "strategy": "HyperOptimizer[greedy,random-greedy]/combo",
                "seed": arguments.seed,
                "metrics": {
                    "search_time_seconds": elapsed,
                    "estimated_flops": int(tree.contraction_cost()),
                    "largest_intermediate_elements": int(tree.max_size()),
                    "estimated_peak_bytes": int(tree.max_size())
                    * workload.element_size,
                    "slice_count": int(tree.nslices),
                    "sliced_labels": list(sliced),
                    "operation_count": len(workload.inputs) - 1,
                    "common_model_estimated_flops": common_metrics[
                        "estimated_flops"
                    ],
                    "common_model_largest_intermediate_elements": common_metrics[
                        "largest_intermediate_elements"
                    ],
                },
                "environment": {
                    "python": platform.python_version(),
                    "isolation": "temporary_target_directory",
                },
                "distribution_semantics": "planning_only",
                "scalability_claim_allowed": False,
            }
    if arguments.output is not None:
        write_result(payload, arguments.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
