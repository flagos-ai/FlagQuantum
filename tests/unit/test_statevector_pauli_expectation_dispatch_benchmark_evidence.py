"""Contracts for reproducible MEAS-002 public-dispatch evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.statevector_pauli_expectation_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    OPERATORS,
    PERFORMANCE_FLOOR,
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

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_ARTIFACT = (
    Path(__file__).parents[2]
    / "benchmarks/results/local/statevector_pauli_expectation_dispatch_a800.json"
)


def _run(host: str, lane: str, *, speedup: float = 1.2) -> dict[str, object]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, amplitudes in SHAPE_MATRIX:
        dispatch = {
            "samples_seconds_per_invocation": samples,
            "median_seconds_per_invocation": statistics.median(samples),
            "peak_memory_bytes": 1024,
            "peak_memory_delta_bytes": 256,
        }
        reference = copy.deepcopy(dispatch)
        reference["samples_seconds_per_invocation"] = [
            sample * speedup for sample in samples
        ]
        reference["median_seconds_per_invocation"] = statistics.median(
            reference["samples_seconds_per_invocation"]
        )
        cases.append(
            {
                "shape": {"batch": batch, "amplitudes": amplitudes},
                "operators": [[wire, axis] for wire, axis in OPERATORS],
                "dtype": "complex64",
                "layout": "contiguous_flat_statevector",
                "route_expected": True,
                "maximum_expectation_absolute_error": 1e-7,
                "expectation_relative_l2_error": 1e-7,
                "maximum_gradient_absolute_error": 1e-7,
                "gradient_relative_l2_error": 1e-7,
                "public_catalog_dispatch_forward": copy.deepcopy(dispatch),
                "public_pytorch_reference_forward": copy.deepcopy(reference),
                "public_catalog_dispatch_forward_backward": copy.deepcopy(dispatch),
                "public_pytorch_reference_forward_backward": copy.deepcopy(reference),
                "public_forward_speedup_over_pytorch": speedup,
                "public_forward_backward_speedup_over_pytorch": speedup,
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": _REVISION,
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": lane,
        "compiler": {
            "distribution": distribution,
            "version": "3.7.1",
            "integration_path": "direct",
            "identity_source": "python_package_metadata",
            "identity_status": "resolved",
        },
        "environment": {
            "gpu": "NVIDIA A800-SXM4-80GB",
            "gpu_total_memory_bytes": 85_000_000_000,
            "cuda_runtime": "13.0",
            "pytorch": "2.13.0",
            "python": "3.12.13",
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": 10,
            "repeats": len(samples),
            "group_size": 10,
            "seed": 261106,
        },
        "performance_floor": PERFORMANCE_FLOOR,
        "cases": cases,
    }


def _matrix(*, speedup: float = 1.2) -> list[dict[str, object]]:
    return [
        _run(host, lane, speedup=speedup)
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


def test_run_validator_enforces_public_performance_floor() -> None:
    with pytest.raises(ValueError, match="misses performance floor"):
        validate_run(_run("jp-a800-171", "stock_triton", speedup=0.99))


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_merge_rejects_mixed_revisions() -> None:
    payloads = _matrix()
    payloads[-1]["source_revision"] = "f" * 40

    with pytest.raises(ValueError, match="same source revision"):
        merge_runs(payloads, required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_selects_default_only_after_every_public_win() -> None:
    payload = merge_runs(
        _matrix(speedup=1.3),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_forward_speedup_range"] == [1.3, 1.3]
    assert payload["public_forward_backward_speedup_range"] == [1.3, 1.3]
    assert payload["public_dispatch_win_on_all_runs"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"


def test_checked_in_a800_evidence_is_canonical_and_selects_default() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert payload["source_revision"] == ("a408866df3e2a795d3a893cc85afab98bb41d42b")
    assert payload["required_hosts"] == ["jp-a800-171", "jp-a800-172"]
    assert payload["required_compiler_lanes"] == ["stock_triton", "flagtree"]
    assert payload["public_dispatch_win_on_all_runs"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"
