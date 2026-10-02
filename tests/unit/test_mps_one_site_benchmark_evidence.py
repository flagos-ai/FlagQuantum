"""Contracts for reproducible MPS-003 benchmark evidence."""

from __future__ import annotations

import copy
import statistics

import pytest

from benchmarks.mps_one_site_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    RATIO_BASELINES,
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
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": 1024,
        "peak_memory_delta_bytes": 256,
    }


def _run(
    host: str,
    lane: str,
    *,
    public_forward_eager: float = 0.9,
    public_forward_compiled: float = 0.8,
    public_backward_eager: float = 0.9,
    public_backward_compiled: float = 0.8,
) -> dict[str, object]:
    cases = []
    for sites, batch, left_bond, right_bond in SHAPE_MATRIX:
        results = {name: _result() for name in RESULT_NAMES}
        multipliers = {
            "direct_pytorch_forward": 1.1,
            "direct_pytorch_forward_backward": 1.2,
            "public_pytorch_eager_forward": public_forward_eager,
            "public_pytorch_compiled_forward": public_forward_compiled,
            "public_pytorch_eager_forward_backward": public_backward_eager,
            "public_pytorch_compiled_forward_backward": public_backward_compiled,
        }
        for name, multiplier in multipliers.items():
            results[name] = _result(multiplier)
        ratios = {
            field: (
                results[baseline]["median_seconds_per_invocation"]
                / results[candidate]["median_seconds_per_invocation"]
            )
            for field, (baseline, candidate) in RATIO_BASELINES.items()
        }
        cases.append(
            {
                "shape": {
                    "sites": sites,
                    "batch": batch,
                    "left_bond": left_bond,
                    "physical_dimension": 2,
                    "right_bond": right_bond,
                },
                "contraction_elements": sites * batch * left_bond * right_bond,
                "dtype": "complex64",
                "layout": "contiguous_site_bucket_and_batched_ry_gates",
                "maximum_forward_absolute_error": 1e-6,
                "maximum_forward_relative_l2_error": 1e-6,
                "maximum_gradient_absolute_error": 1e-6,
                "maximum_gradient_relative_l2_error": 1e-6,
                **results,
                **ratios,
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
            "repeats": 3,
            "group_size": 4,
            "seed": 270003,
            "directions": ["forward", "forward_backward"],
        },
        "cases": cases,
    }


def _matrix(
    *,
    public_forward_eager: float = 0.9,
    public_forward_compiled: float = 0.8,
    public_backward_eager: float = 0.9,
    public_backward_compiled: float = 0.8,
) -> list[dict[str, object]]:
    return [
        _run(
            host,
            lane,
            public_forward_eager=public_forward_eager,
            public_forward_compiled=public_forward_compiled,
            public_backward_eager=public_backward_eager,
            public_backward_compiled=public_backward_compiled,
        )
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_forward_and_backward_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_run_validator_recomputes_each_median(result_name: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][result_name]["median_seconds_per_invocation"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


@pytest.mark.parametrize("field", tuple(RATIO_BASELINES))
def test_run_validator_recomputes_each_speedup(field: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][field] = 9.0

    with pytest.raises(ValueError, match=f"{field} is not reproducible"):
        validate_run(payload)


def test_run_validator_recomputes_contraction_elements() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["contraction_elements"] = 1

    with pytest.raises(ValueError, match="contraction_elements is not reproducible"):
        validate_run(payload)


def test_run_validator_requires_both_derivative_directions() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["measurement"]["directions"] = ["forward"]

    with pytest.raises(ValueError, match="forward and backward"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_retains_opt_in_when_one_public_direction_loses() -> None:
    payload = merge_runs(
        _matrix(
            public_forward_eager=1.1,
            public_forward_compiled=1.2,
            public_backward_eager=1.1,
            public_backward_compiled=0.9,
        ),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_forward_win_over_eager_on_all_cases"]
    assert payload["public_forward_win_over_compiled_on_all_cases"]
    assert payload["public_forward_backward_win_over_eager_on_all_cases"]
    assert not payload["public_forward_backward_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "retain_opt_in"


def test_aggregate_allows_default_only_after_all_public_wins() -> None:
    payload = merge_runs(
        _matrix(
            public_forward_eager=1.1,
            public_forward_compiled=1.2,
            public_backward_eager=1.1,
            public_backward_compiled=1.2,
        ),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_forward_win_over_eager_on_all_cases"]
    assert payload["public_forward_win_over_compiled_on_all_cases"]
    assert payload["public_forward_backward_win_over_eager_on_all_cases"]
    assert payload["public_forward_backward_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"


def test_aggregate_validation_rejects_noncanonical_decision() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))
    changed = copy.deepcopy(payload)
    changed["dispatch_selection_decision"] = "eligible_for_default"

    with pytest.raises(ValueError, match="canonical merge"):
        validate_evidence(changed)
