"""Run the native FlagQuantum planning baseline on a public TN workload."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from benchmarks.runners.tn.tn_gap_common import (
    RESULT_SCHEMA,
    load_workload,
    write_result,
)
from flagquantum.simulation.tensor_models import (
    TensorNetworkContractionPlan,
    TensorNetworkNode,
)
from flagquantum.version import __version__


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=(
            "greedy",
            "memory_greedy",
            "quality_greedy",
            "quality_multistart",
            "quality_reconfigured",
            "beam",
        ),
        default="quality_greedy",
    )
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--search-budget-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    workload = load_workload(arguments.workload)
    dtype = getattr(torch, workload.dtype)
    nodes = tuple(
        TensorNetworkNode(
            tensor=torch.empty(shape, dtype=dtype, device="meta"),
            labels=labels,
            name=f"input:{index}",
        )
        for index, (labels, shape) in enumerate(
            zip(workload.inputs, workload.shapes, strict=True)
        )
    )
    plan = TensorNetworkContractionPlan(
        n_wires=0,
        bsz=1,
        nodes=nodes,
        output_labels=workload.output,
        path=(),
    )
    started = time.perf_counter()
    try:
        profile = plan.contraction_profile(
            arguments.strategy,
            beam_width=arguments.beam_width,
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        payload = {
            "schema_version": RESULT_SCHEMA,
            "backend": "flagquantum",
            "backend_version": __version__,
            "workload_identity": workload.identity,
            "workload_name": workload.name,
            "comparison_scope": "planning",
            "search_budget_seconds": arguments.search_budget_seconds,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "metrics": {},
            "environment": {"python": platform.python_version()},
            "distribution_semantics": "planning_only",
            "scalability_claim_allowed": False,
        }
    else:
        elapsed = time.perf_counter() - started
        budget_compliant = (
            arguments.search_budget_seconds <= 0
            or elapsed <= max(0.05, arguments.search_budget_seconds * 1.10)
        )
        payload = {
            "schema_version": RESULT_SCHEMA,
            "backend": "flagquantum",
            "backend_version": __version__,
            "workload_identity": workload.identity,
            "workload_name": workload.name,
            "comparison_scope": "planning",
            "search_budget_seconds": arguments.search_budget_seconds,
            "status": "completed",
            "search_budget_compliant": budget_compliant,
            "strategy": arguments.strategy,
            "metrics": {
                "search_time_seconds": elapsed,
                "estimated_flops": int(profile.estimated_cost),
                "largest_intermediate_elements": int(profile.peak_size),
                "estimated_peak_bytes": int(profile.peak_size)
                * workload.element_size,
                "slice_count": int(profile.n_slices),
                "sliced_labels": list(profile.sliced_labels),
                "total_intermediate_elements": int(
                    profile.total_intermediate_size
                ),
                "operation_count": int(profile.n_steps),
                "common_model_estimated_flops": int(profile.estimated_cost),
                "common_model_largest_intermediate_elements": int(
                    profile.peak_size
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            "distribution_semantics": "planning_only",
            "scalability_claim_allowed": False,
        }
    if arguments.output is not None:
        write_result(payload, arguments.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
