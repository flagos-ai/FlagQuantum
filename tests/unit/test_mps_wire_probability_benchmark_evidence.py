"""Contracts for reproducible MPS-007 benchmark evidence."""

from __future__ import annotations

import copy
import statistics

import pytest

from benchmarks.mps_wire_probability_dispatch import (
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


def _run(host: str, lane: str, *, public_speedup: float = 0.9) -> dict[str, object]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, left_bond, right_bond in SHAPE_MATRIX:
        result = {
            "samples_seconds_per_invocation": samples,
            "median_seconds_per_invocation": statistics.median(samples),
            "peak_memory_bytes": 1024,
            "peak_memory_delta_bytes": 256,
        }
        public_dispatch = copy.deepcopy(result)
        public_reference = copy.deepcopy(result)
        public_reference["samples_seconds_per_invocation"] = [
            sample * public_speedup for sample in samples
        ]
        public_reference["median_seconds_per_invocation"] = statistics.median(
            public_reference["samples_seconds_per_invocation"]
        )
        observed_public_speedup = (
            public_reference["median_seconds_per_invocation"]
            / public_dispatch["median_seconds_per_invocation"]
        )
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "left_bond": left_bond,
                    "physical_dimension": 2,
                    "right_bond": right_bond,
                },
                "dtype": "complex64",
                "layout": "contiguous_mps_site_tensor",
                "maximum_absolute_error": 1e-7,
                "relative_l2_error": 1e-7,
                "direct_kernel_wrapper": copy.deepcopy(result),
                "direct_pytorch_reduction": copy.deepcopy(result),
                "public_catalog_dispatch": public_dispatch,
                "public_pytorch_reference": public_reference,
                "direct_kernel_speedup_over_pytorch": 1.0,
                "public_dispatch_speedup_over_pytorch": observed_public_speedup,
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
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": 2,
            "repeats": len(samples),
            "group_size": 4,
            "seed": 261001,
        },
        "cases": cases,
    }


def _matrix(*, public_speedup: float = 0.9) -> list[dict[str, object]]:
    return [
        _run(host, lane, public_speedup=public_speedup)
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


def test_run_validator_recomputes_public_speedup() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["public_dispatch_speedup_over_pytorch"] = 9.0

    with pytest.raises(ValueError, match="public speedup is not reproducible"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_records_opt_in_decision_without_public_win() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))

    validate_evidence(payload)
    assert not payload["public_dispatch_win_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "retain_opt_in"


def test_aggregate_allows_evidence_to_select_a_measured_public_win() -> None:
    payload = merge_runs(
        _matrix(public_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_dispatch_win_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"
