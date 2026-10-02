"""Contracts for reproducible MPS-001 benchmark evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.mps_two_site_dispatch import (
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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "mps_two_site_dispatch_a800.json"
)


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
    catalog_forward_eager: float = 0.9,
    catalog_forward_compiled: float = 0.8,
    catalog_backward_eager: float = 0.9,
    catalog_backward_compiled: float = 0.8,
) -> dict[str, object]:
    cases = []
    for batch, left_bond, middle_bond, right_bond, batched_gate in SHAPE_MATRIX:
        results = {name: _result() for name in RESULT_NAMES}
        multipliers = {
            "direct_pytorch_forward": 1.1,
            "direct_pytorch_forward_backward": 1.2,
            "eager_reference_forward": catalog_forward_eager,
            "compiled_reference_forward": catalog_forward_compiled,
            "eager_reference_forward_backward": catalog_backward_eager,
            "compiled_reference_forward_backward": catalog_backward_compiled,
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
                    "batch": batch,
                    "left_bond": left_bond,
                    "physical_dimension": 2,
                    "middle_bond": middle_bond,
                    "right_bond": right_bond,
                    "batched_gate": batched_gate,
                },
                "contraction_elements": (batch * left_bond * middle_bond * right_bond),
                "runtime_forward_eligible": (
                    batch * left_bond * middle_bond * right_bond >= 2**12
                ),
                "runtime_backward_eligible": (
                    batch * left_bond * middle_bond * right_bond >= 2**18
                ),
                "dtype": "complex64",
                "layout": "contiguous_mps_two_site_and_shared_or_batched_gate",
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
            "seed": 270001,
            "directions": ["forward", "forward_backward"],
        },
        "cases": cases,
    }


def _matrix(
    *,
    catalog_forward_eager: float = 0.9,
    catalog_forward_compiled: float = 0.8,
    catalog_backward_eager: float = 0.9,
    catalog_backward_compiled: float = 0.8,
) -> list[dict[str, object]]:
    return [
        _run(
            host,
            lane,
            catalog_forward_eager=catalog_forward_eager,
            catalog_forward_compiled=catalog_forward_compiled,
            catalog_backward_eager=catalog_backward_eager,
            catalog_backward_compiled=catalog_backward_compiled,
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


@pytest.mark.parametrize(
    "field", ("runtime_forward_eligible", "runtime_backward_eligible")
)
def test_run_validator_recomputes_runtime_eligibility(field: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][field] = not payload["cases"][0][field]

    with pytest.raises(ValueError, match=f"{field} is not reproducible"):
        validate_run(payload)


def test_run_validator_requires_both_derivative_directions() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["measurement"]["directions"] = ["forward"]

    with pytest.raises(ValueError, match="forward and backward"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_retains_opt_in_when_one_catalog_direction_loses() -> None:
    payload = merge_runs(
        _matrix(
            catalog_forward_eager=1.1,
            catalog_forward_compiled=1.2,
            catalog_backward_eager=1.1,
            catalog_backward_compiled=0.9,
        ),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["catalog_forward_win_over_eager_on_all_cases"]
    assert payload["catalog_forward_win_over_compiled_on_all_cases"]
    assert payload["catalog_forward_backward_win_over_eager_on_all_cases"]
    assert not payload["catalog_forward_backward_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "retain_opt_in"


def test_aggregate_retains_opt_in_without_end_to_end_factorization_evidence() -> None:
    payload = merge_runs(
        _matrix(
            catalog_forward_eager=1.1,
            catalog_forward_compiled=1.2,
            catalog_backward_eager=1.1,
            catalog_backward_compiled=1.2,
        ),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["catalog_forward_win_over_eager_on_all_cases"]
    assert payload["catalog_forward_win_over_compiled_on_all_cases"]
    assert payload["catalog_forward_backward_win_over_eager_on_all_cases"]
    assert payload["catalog_forward_backward_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "retain_opt_in"
    assert payload["default_dispatch_blockers"] == [
        "catalog-route evidence excludes the downstream MPS factorization"
    ]


def test_aggregate_validation_rejects_noncanonical_decision() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))
    changed = copy.deepcopy(payload)
    changed["dispatch_selection_decision"] = "eligible_for_default"

    with pytest.raises(ValueError, match="dispatch_selection_decision"):
        validate_evidence(changed)


def test_checked_in_a800_evidence_is_canonical_and_retains_opt_in() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert not payload["direct_forward_win_on_all_cases"]
    assert not payload["direct_forward_backward_win_on_all_cases"]
    assert not payload["catalog_forward_win_over_eager_on_all_cases"]
    assert payload["catalog_forward_win_over_compiled_on_all_cases"]
    assert not payload["catalog_forward_backward_win_over_eager_on_all_cases"]
    assert payload["catalog_forward_backward_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "retain_opt_in"
    assert payload["default_dispatch_blockers"] == [
        "catalog-route evidence excludes the downstream MPS factorization"
    ]
