"""Contracts for reproducible NUM-002 benchmark evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.tn_layout_contraction import (
    COMPILER_LANES,
    EQUATION,
    FALLBACK_MAXIMUM_OVERHEAD_SECONDS,
    FALLBACK_MINIMUM_SPEEDUP,
    IMPLEMENTATION_ID,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SELECTED_LOGICAL_BMM_SHAPES,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _fallback_within_overhead_budget,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "tn_layout_contraction_a800.json"
)


def _measurement(scale: float = 1.0) -> dict[str, object]:
    samples = [value * scale for value in (1e-4, 2e-4, 3e-4)]
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": 2048,
        "peak_memory_delta_bytes": 512,
    }


def _run(
    host: str,
    lane: str,
    *,
    selected_public_speedup: float = 0.9,
    fallback_public_speedup: float = 1.0,
    revision: str = "0123456789abcdef",
) -> dict[str, object]:
    cases = []
    for batch, rows, reduction_left, reduction_right, columns in SHAPE_MATRIX:
        logical_shape = (batch, rows, reduction_left * reduction_right, columns)
        selected = logical_shape in SELECTED_LOGICAL_BMM_SHAPES
        public_speedup = (
            selected_public_speedup if selected else fallback_public_speedup
        )
        direct_forward = _measurement()
        materialized_forward = _measurement(1.1)
        native_forward = _measurement(public_speedup)
        public_dispatch = _measurement()
        direct_training = _measurement()
        native_training = _measurement(public_speedup)
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "rows": rows,
                    "reduction_left": reduction_left,
                    "reduction_right": reduction_right,
                    "columns": columns,
                },
                "logical_bmm_shape": [
                    batch,
                    rows,
                    reduction_left * reduction_right,
                    columns,
                ],
                "dtype": "complex64",
                "layout": "explicit_strided_batch",
                "equation": EQUATION,
                "public_dispatch_route": (
                    "catalog_kernel" if selected else "native_einsum"
                ),
                "maximum_absolute_error": 1e-5,
                "relative_l2_error": 1e-6,
                "maximum_gradient_absolute_error": 2e-5,
                "gradient_relative_l2_error": 2e-6,
                "direct_layout_forward": direct_forward,
                "materialized_torch_bmm_forward": materialized_forward,
                "native_einsum_forward": native_forward,
                "public_catalog_dispatch": public_dispatch,
                "direct_layout_forward_backward": direct_training,
                "native_einsum_forward_backward": native_training,
                "direct_forward_speedup_over_native": (
                    native_forward["median_seconds_per_invocation"]
                    / direct_forward["median_seconds_per_invocation"]
                ),
                "direct_forward_speedup_over_materialized": (
                    materialized_forward["median_seconds_per_invocation"]
                    / direct_forward["median_seconds_per_invocation"]
                ),
                "public_dispatch_speedup_over_native": (
                    native_forward["median_seconds_per_invocation"]
                    / public_dispatch["median_seconds_per_invocation"]
                ),
                "direct_training_speedup_over_native": (
                    native_training["median_seconds_per_invocation"]
                    / direct_training["median_seconds_per_invocation"]
                ),
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": revision,
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": lane,
        "compiler": {
            "distribution": distribution,
            "version": "3.0.0",
            "integration_path": "direct" if lane == "stock_triton" else "flagtree",
            "identity_source": "python_package_metadata",
            "identity_status": "resolved",
        },
        "environment": {
            "gpu": "NVIDIA A800-SXM4-80GB",
            "gpu_total_memory_bytes": 85_000_000_000,
            "cuda_runtime": "12.8",
            "pytorch": "2.7.1",
            "python": "3.12.11",
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": 2,
            "repeats": 3,
            "group_size": 4,
            "seed": 261003,
        },
        "cases": cases,
    }


def _matrix(
    *,
    selected_public_speedup: float = 0.9,
    fallback_public_speedup: float = 1.0,
    revision: str = "0123456789abcdef",
) -> list[dict[str, object]]:
    return [
        _run(
            host,
            lane,
            selected_public_speedup=selected_public_speedup,
            fallback_public_speedup=fallback_public_speedup,
            revision=revision,
        )
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_run_validator_recomputes_each_median(result_name: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][result_name]["median_seconds_per_invocation"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


def test_run_validator_rejects_gradient_error_outside_contract() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["gradient_relative_l2_error"] = 1.0

    with pytest.raises(ValueError, match="gradient_relative_l2_error"):
        validate_run(payload)


def test_run_validator_recomputes_public_speedup() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["public_dispatch_speedup_over_native"] = 9.0

    with pytest.raises(ValueError, match="not reproducible"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_merge_requires_one_source_revision() -> None:
    matrix = _matrix()
    matrix[-1] = _run("jp-a800-172", "flagtree", revision="different")

    with pytest.raises(ValueError, match="same source revision"):
        merge_runs(matrix, required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_requests_policy_review_without_selected_kernel_win() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))

    validate_evidence(payload)
    assert not payload["selected_kernel_win_on_all_cases"]
    assert payload["fallback_within_overhead_budget_on_all_cases"]
    assert payload["dispatch_evidence_decision"] == "revisit_current_policy"


def test_aggregate_retains_policy_with_selected_win_and_safe_fallback() -> None:
    payload = merge_runs(
        _matrix(selected_public_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["selected_kernel_win_on_all_cases"]
    assert payload["fallback_within_overhead_budget_on_all_cases"]
    assert payload["dispatch_evidence_decision"] == "retain_current_policy"


def test_aggregate_requests_policy_review_for_fallback_regression() -> None:
    payload = merge_runs(
        _matrix(fallback_public_speedup=FALLBACK_MINIMUM_SPEEDUP - 0.01),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert not payload["fallback_within_overhead_budget_on_all_cases"]
    assert payload["dispatch_evidence_decision"] == "revisit_current_policy"


def test_fallback_budget_accepts_bounded_absolute_wrapper_overhead() -> None:
    native_seconds = 56e-6
    public_seconds = native_seconds + FALLBACK_MAXIMUM_OVERHEAD_SECONDS
    case = {
        "native_einsum_forward": {
            "median_seconds_per_invocation": native_seconds,
        },
        "public_catalog_dispatch": {
            "median_seconds_per_invocation": public_seconds,
        },
        "public_dispatch_speedup_over_native": native_seconds / public_seconds,
    }

    assert case["public_dispatch_speedup_over_native"] < FALLBACK_MINIMUM_SPEEDUP
    assert _fallback_within_overhead_budget(case)


def test_evidence_validator_rejects_noncanonical_summary() -> None:
    payload = merge_runs(
        _matrix(selected_public_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )
    changed = copy.deepcopy(payload)
    changed["dispatch_evidence_decision"] = "revisit_current_policy"

    with pytest.raises(ValueError, match="canonical merge"):
        validate_evidence(changed)


def test_checked_in_a800_evidence_is_canonical_and_retains_narrow_policy() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert not payload["direct_forward_win_on_all_cases"]
    assert not payload["direct_training_win_on_all_cases"]
    assert payload["selected_kernel_win_on_all_cases"]
    assert payload["fallback_within_overhead_budget_on_all_cases"]
    assert payload["dispatch_evidence_decision"] == "retain_current_policy"
