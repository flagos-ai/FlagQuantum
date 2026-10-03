"""Contracts for reproducible MEAS-003 A800 evidence."""

from __future__ import annotations

import copy
import statistics

import pytest

from benchmarks.statevector_marginal_probability_kernel import (
    IMPLEMENTATION_ID,
    PERFORMANCE_FLOOR,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    merge_runs,
    validate_run,
)

pytestmark = pytest.mark.unit

_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _run(host: str, *, speedup: float = 1.2) -> dict[str, object]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, amplitudes, qubits in SHAPE_MATRIX:
        triton_result = {
            "samples_seconds_per_invocation": samples,
            "median_seconds_per_invocation": statistics.median(samples),
            "peak_memory_bytes": 1024,
            "peak_memory_delta_bytes": 256,
        }
        pytorch_result = copy.deepcopy(triton_result)
        pytorch_result["samples_seconds_per_invocation"] = [
            sample * speedup for sample in samples
        ]
        pytorch_result["median_seconds_per_invocation"] = statistics.median(
            pytorch_result["samples_seconds_per_invocation"]
        )
        cases.append(
            {
                "shape": {"batch": batch, "amplitudes": amplitudes},
                "qubits": list(qubits),
                "dtype": "complex64",
                "layout": "contiguous_flat_statevector",
                "maximum_probability_absolute_error": 1e-7,
                "probability_relative_l2_error": 1e-7,
                "maximum_gradient_absolute_error": 1e-7,
                "gradient_relative_l2_error": 1e-7,
                "triton_forward": copy.deepcopy(triton_result),
                "pytorch_forward": copy.deepcopy(pytorch_result),
                "triton_forward_backward": copy.deepcopy(triton_result),
                "pytorch_forward_backward": copy.deepcopy(pytorch_result),
                "forward_speedup_over_pytorch": speedup,
                "forward_backward_speedup_over_pytorch": speedup,
            }
        )
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": _REVISION,
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": "triton",
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
            "seed": 261004,
        },
        "performance_floor": PERFORMANCE_FLOOR,
        "cases": cases,
    }


def _matrix(*, speedup: float = 1.2) -> list[dict[str, object]]:
    return [_run(host, speedup=speedup) for host in ("jp-a800-171", "jp-a800-172")]


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171"))


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_run_validator_recomputes_each_median(result_name: str) -> None:
    payload = _run("jp-a800-171")
    payload["cases"][0][result_name]["median_seconds_per_invocation"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


def test_run_validator_requires_canonical_matrix() -> None:
    payload = _run("jp-a800-171")
    payload["cases"][0]["qubits"] = [0]

    with pytest.raises(ValueError, match="fixed MEAS-003 shape matrix"):
        validate_run(payload)


def test_run_validator_requires_full_revision() -> None:
    payload = _run("jp-a800-171")
    payload["source_revision"] = _REVISION[:12]

    with pytest.raises(ValueError, match="full lowercase Git revision"):
        validate_run(payload)


def test_run_validator_enforces_performance_floor() -> None:
    payload = _run("jp-a800-171", speedup=0.99)

    with pytest.raises(ValueError, match="misses performance floor"):
        validate_run(payload)


def test_merge_requires_both_hosts() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_merge_rejects_mixed_revisions() -> None:
    payloads = _matrix()
    payloads[1]["source_revision"] = "f" * 40

    with pytest.raises(ValueError, match="same source revision"):
        merge_runs(payloads, required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_records_speedup_ranges_and_floor() -> None:
    payload = merge_runs(
        _matrix(speedup=1.3),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    assert payload["forward_speedup_range"] == [1.3, 1.3]
    assert payload["forward_backward_speedup_range"] == [1.3, 1.3]
    assert payload["performance_floor"] == 1.0
    assert payload["implementation_decision"] == "retain_experimental"
