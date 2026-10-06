"""Contracts for reproducible FlagTree TLE control-transport evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.flagtree_tle_control_transport import (
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTICS,
    SHAPE_MATRIX,
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
    / "flagtree_tle_control_transport_a800.json"
)


def _timing(*, scale: float = 1.0) -> dict[str, object]:
    samples = [scale * value for value in (1e-4, 2e-4, 3e-4)]
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": 1024,
        "peak_memory_delta_bytes": 256,
    }


def _operation(*, tle_speedup: float) -> dict[str, object]:
    tle = _timing()
    shared = _timing(scale=tle_speedup)
    pytorch = _timing(scale=2.0)
    tle_median = tle["median_seconds_per_invocation"]
    shared_median = shared["median_seconds_per_invocation"]
    pytorch_median = pytorch["median_seconds_per_invocation"]
    return {
        "flagtree_tle": tle,
        "shared_triton": shared,
        "pytorch_reference": pytorch,
        "tle_speedup_over_shared_triton": shared_median / tle_median,
        "tle_speedup_over_pytorch": pytorch_median / tle_median,
        "shared_triton_speedup_over_pytorch": pytorch_median / shared_median,
    }


def _run(host: str, *, tle_speedup: float = 0.9) -> dict[str, object]:
    cases = []
    for (
        batch,
        amplitudes,
        bit_position,
        compressed_start,
        compressed_end,
    ) in SHAPE_MATRIX:
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "amplitudes_per_batch": amplitudes,
                    "bit_position": bit_position,
                    "compressed_start": compressed_start,
                    "compressed_end": compressed_end,
                },
                "dtype": "complex64",
                "layout": "contiguous_flat_statevector_to_packed_subspace",
                "maximum_absolute_error": 0.0,
                "relative_l2_error": 0.0,
                "operations": {
                    name: _operation(tle_speedup=tle_speedup) for name in SEMANTICS
                },
            }
        )
    return {
        "schema": RUN_SCHEMA,
        "semantics": SEMANTICS,
        "source_revision": "2e46d88e7",
        "runner": RUNNER,
        "command": f"python3 {RUNNER} run",
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
            "repeats": 3,
            "group_size": 4,
            "seed": 261006,
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
    payload["cases"][0]["operations"]["pack"][result_name][
        "median_seconds_per_invocation"
    ] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


def test_merge_requires_both_hosts_exactly_once() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_retains_explicit_without_cross_matrix_win() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))

    validate_evidence(payload)
    assert not payload["tle_win_over_shared_triton_on_all_operations_and_cases"]
    assert payload["provider_selection_decision"] == "retain_explicit"


def test_aggregate_allows_cross_matrix_win_to_enter_dispatch_study() -> None:
    payload = merge_runs(
        _matrix(tle_speedup=1.1),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["tle_win_over_shared_triton_on_all_operations_and_cases"]
    assert payload["provider_selection_decision"] == "eligible_for_dispatch_study"


def test_checked_in_a800_evidence_is_canonical() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert not payload["tle_win_over_shared_triton_on_all_operations_and_cases"]
    assert payload["provider_selection_decision"] == "retain_explicit"


def test_validator_rejects_non_flagtree_compiler() -> None:
    payload = copy.deepcopy(_run("jp-a800-171"))
    payload["compiler"]["distribution"] = "triton"

    with pytest.raises(ValueError, match="resolved FlagTree CUDA compiler"):
        validate_run(payload)
