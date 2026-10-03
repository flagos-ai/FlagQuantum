"""Contracts for reproducible FlagTree TLE sharded-adjoint evidence."""

from __future__ import annotations

import copy
import statistics

import pytest

from benchmarks.flagtree_tle_sharded_adjoint import (
    BASELINE_IMPLEMENTATION_ID,
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


def _run(host: str, *, tle_speedup: float = 0.9) -> dict[str, object]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, elements_per_chunk, rank_basis in SHAPE_MATRIX:
        tle = {
            "samples_seconds_per_invocation": samples,
            "median_seconds_per_invocation": statistics.median(samples),
            "peak_memory_bytes": 1024,
            "peak_memory_delta_bytes": 256,
        }
        shared = copy.deepcopy(tle)
        shared["samples_seconds_per_invocation"] = [
            sample * tle_speedup for sample in samples
        ]
        shared["median_seconds_per_invocation"] = statistics.median(
            shared["samples_seconds_per_invocation"]
        )
        pytorch = copy.deepcopy(tle)
        pytorch["samples_seconds_per_invocation"] = [sample * 2 for sample in samples]
        pytorch["median_seconds_per_invocation"] = statistics.median(
            pytorch["samples_seconds_per_invocation"]
        )
        tle_median = tle["median_seconds_per_invocation"]
        shared_median = shared["median_seconds_per_invocation"]
        pytorch_median = pytorch["median_seconds_per_invocation"]
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "elements_per_chunk": elements_per_chunk,
                    "rank_basis": rank_basis,
                },
                "dtype": "complex64",
                "layout": "contiguous_sharded_statevector",
                "next_adjoint_maximum_absolute_error": 1e-7,
                "next_adjoint_relative_l2_error": 1e-7,
                "gradient_reference": 12.0,
                "gradient_absolute_error": 1e-4,
                "gradient_relative_error": 1e-5,
                "flagtree_tle": tle,
                "shared_triton": shared,
                "pytorch_reference": pytorch,
                "tle_speedup_over_shared_triton": shared_median / tle_median,
                "tle_speedup_over_pytorch": pytorch_median / tle_median,
                "shared_triton_speedup_over_pytorch": (pytorch_median / shared_median),
            }
        )
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "baseline_implementation_id": BASELINE_IMPLEMENTATION_ID,
        "source_revision": "184c9686e49e472a97988d931b6a7469710165ba",
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": "flagtree",
            "version": "0.7.0",
            "triton_api_version": "3.6.0",
            "integration_path": "flagtree",
            "backend": "cuda",
            "identity_source": "python_package_metadata",
            "identity_status": "resolved",
        },
        "environment": {
            "gpu": "NVIDIA A800-SXM4-80GB",
            "gpu_total_memory_bytes": 85_000_000_000,
            "cuda_runtime": "12.8",
            "pytorch": "2.7.1",
            "python": "3.12.3",
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": 2,
            "repeats": len(samples),
            "group_size": 4,
            "seed": 261007,
        },
        "cases": cases,
    }


def _matrix(*, tle_speedup: float = 0.9) -> list[dict[str, object]]:
    return [
        _run("jp-a800-171", tle_speedup=tle_speedup),
        _run("jp-a800-172", tle_speedup=tle_speedup),
    ]


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171"))


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_run_validator_recomputes_each_median(result_name: str) -> None:
    payload = _run("jp-a800-171")
    payload["cases"][0][result_name]["median_seconds_per_invocation"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


def test_run_validator_enforces_gradient_error_contract() -> None:
    payload = _run("jp-a800-171")
    payload["cases"][0]["gradient_absolute_error"] = 1.0

    with pytest.raises(ValueError, match="gradient error tolerance"):
        validate_run(payload)


def test_merge_requires_both_hosts_exactly_once() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_retains_explicit_provider_without_cross_matrix_win() -> None:
    payload = merge_runs(
        _matrix(),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert not payload["tle_win_over_shared_triton_on_all_cases"]
    assert payload["provider_selection_decision"] == "retain_explicit"


def test_aggregate_allows_a_cross_matrix_win_to_enter_dispatch_study() -> None:
    payload = merge_runs(
        _matrix(tle_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["tle_win_over_shared_triton_on_all_cases"]
    assert payload["provider_selection_decision"] == "eligible_for_dispatch_study"
