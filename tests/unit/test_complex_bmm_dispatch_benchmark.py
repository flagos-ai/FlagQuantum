"""Contracts for reproducible NUM-001 complex BMM evidence."""

from __future__ import annotations

import statistics

import pytest

from benchmarks.complex_bmm_dispatch import (
    COMPILER_LANES,
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


def _result(multiplier: float = 1.0) -> dict[str, object]:
    samples = [1e-4 * multiplier, 2e-4 * multiplier, 3e-4 * multiplier]
    return {
        "median_seconds_per_invocation": statistics.median(samples),
        "samples_seconds_per_invocation": samples,
        "peak_memory_delta_bytes": 1024,
    }


def _run(host: str, lane: str, *, triton_multiplier: float = 0.8) -> dict[str, object]:
    cases = []
    for batch, rows, reduction, columns in SHAPE_MATRIX:
        pytorch_forward = _result()
        triton_forward = _result(triton_multiplier)
        pytorch_training = _result()
        triton_training = _result(triton_multiplier)
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "rows": rows,
                    "reduction": reduction,
                    "columns": columns,
                },
                "dtype": "complex64",
                "layout": "contiguous_batched_matrix",
                "workload_origin": "mps_canonical_transfer_absorption",
                "maximum_forward_absolute_error": 1e-6,
                "forward_relative_l2_error": 1e-6,
                "maximum_gradient_absolute_error": 1e-6,
                "gradient_relative_l2_error": 1e-6,
                "pytorch_bmm_forward": pytorch_forward,
                "triton_forward": triton_forward,
                "pytorch_bmm_forward_backward": pytorch_training,
                "triton_forward_backward": triton_training,
                "triton_forward_speedup_over_pytorch_bmm": (
                    pytorch_forward["median_seconds_per_invocation"]
                    / triton_forward["median_seconds_per_invocation"]
                ),
                "triton_training_speedup_over_pytorch_bmm": (
                    pytorch_training["median_seconds_per_invocation"]
                    / triton_training["median_seconds_per_invocation"]
                ),
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "compiler_lane": lane,
        "compiler": {
            "distribution": distribution,
            "version": "3.7.0",
            "integration_path": "direct" if lane == "stock_triton" else "flagtree",
            "identity_source": "python_package_metadata",
            "identity_status": "resolved",
        },
        "environment": {
            "gpu": "NVIDIA A800-SXM4-80GB",
            "gpu_total_memory_bytes": 85_000_000_000,
            "cuda_runtime": "13.0",
            "pytorch": "2.13.0",
            "python": "3.12.14",
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "ordering": "counterbalanced baseline/candidate by repeat parity",
            "warmup": 2,
            "repeats": 3,
            "group_size": 4,
            "seed": 271001,
        },
        "cases": cases,
    }


def _matrix(*, triton_multiplier: float = 0.8) -> list[dict[str, object]]:
    return [
        _run(host, lane, triton_multiplier=triton_multiplier)
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_bmm_forward_and_backward_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_run_validator_recomputes_each_median(result_name: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][result_name]["median_seconds_per_invocation"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


@pytest.mark.parametrize(
    "field",
    (
        "triton_forward_speedup_over_pytorch_bmm",
        "triton_training_speedup_over_pytorch_bmm",
    ),
)
def test_run_validator_recomputes_speedups(field: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][field] = 9.0

    with pytest.raises(ValueError, match="speedup is not reproducible"):
        validate_run(payload)


def test_merge_requires_full_host_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="every required host/compiler lane"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_authorizes_dispatch_only_after_forward_and_training_wins() -> None:
    winning = merge_runs(
        _matrix(triton_multiplier=0.8),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )
    validate_evidence(winning)
    assert winning["runtime_dispatch_authorized"] is True

    losing = merge_runs(
        _matrix(triton_multiplier=1.2),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )
    validate_evidence(losing)
    assert losing["runtime_dispatch_authorized"] is False


def test_aggregate_validator_rejects_edited_decision() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))
    payload["runtime_dispatch_authorized"] = False

    with pytest.raises(ValueError, match="canonical merge"):
        validate_evidence(payload)
