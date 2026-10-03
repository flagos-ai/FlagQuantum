"""Contracts for reproducible NUM-002 benchmark evidence."""

from __future__ import annotations

import copy
import statistics

import pytest

from benchmarks.tn_layout_contraction import (
    COMPILER_LANES,
    EQUATION,
    IMPLEMENTATION_ID,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit


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
    public_speedup: float = 0.9,
    revision: str = "0123456789abcdef",
) -> dict[str, object]:
    cases = []
    for batch, rows, reduction_left, reduction_right, columns in SHAPE_MATRIX:
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
    *, public_speedup: float = 0.9, revision: str = "0123456789abcdef"
) -> list[dict[str, object]]:
    return [
        _run(host, lane, public_speedup=public_speedup, revision=revision)
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


def test_aggregate_requests_policy_review_without_public_win() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))

    validate_evidence(payload)
    assert not payload["public_dispatch_win_on_all_cases"]
    assert payload["dispatch_evidence_decision"] == "revisit_current_policy"


def test_aggregate_retains_policy_only_with_cross_matrix_public_win() -> None:
    payload = merge_runs(
        _matrix(public_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_dispatch_win_on_all_cases"]
    assert payload["dispatch_evidence_decision"] == "retain_current_policy"


def test_evidence_validator_rejects_noncanonical_summary() -> None:
    payload = merge_runs(
        _matrix(public_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )
    changed = copy.deepcopy(payload)
    changed["dispatch_evidence_decision"] = "revisit_current_policy"

    with pytest.raises(ValueError, match="canonical merge"):
        validate_evidence(changed)
