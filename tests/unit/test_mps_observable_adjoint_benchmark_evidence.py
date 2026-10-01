"""Contracts for reproducible MPS-006 benchmark evidence."""

from __future__ import annotations

import copy
import statistics

import pytest

from benchmarks.mps_observable_adjoint_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit


def _run(host: str, lane: str) -> dict[str, object]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, left_bond, right_bond in SHAPE_MATRIX:
        result = {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "peak_memory_bytes": 1024,
            "peak_memory_delta_bytes": 256,
        }
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "left_bond": left_bond,
                    "physical_dimension": 2,
                    "right_bond": right_bond,
                },
                "dtype": "complex64",
                "layout": "contiguous_mps_local_observable",
                "maximum_absolute_error": 1e-5,
                "relative_l2_error": 1e-5,
                "catalog_dispatch": copy.deepcopy(result),
                "pytorch_autograd": copy.deepcopy(result),
                "speedup_over_pytorch_autograd": 1.0,
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": "0123456789abcdef",
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
            "synchronization": "torch.cuda.synchronize after every invocation",
            "warmup": 2,
            "repeats": len(samples),
            "seed": 260930,
        },
        "cases": cases,
    }


def _matrix() -> list[dict[str, object]]:
    return [
        _run(host, lane)
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


def test_run_validator_recomputes_median_from_raw_samples() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["catalog_dispatch"]["median_seconds"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_round_trip_is_canonical() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))

    validate_evidence(payload)
