"""Contracts for reproducible MPS-002 benchmark evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.mps_projected_two_site_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    RATIO_BASELINES,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _measure_counterbalanced,
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
    / "mps_projected_two_site_dispatch_a800.json"
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
    projected_eager: float = 1.1,
    materialized: float = 1.2,
    compiled_projected: float = 0.9,
    eager_factorization: float = 1.1,
) -> dict[str, object]:
    cases = []
    for batch, left_bond, middle_bond, right_bond, rank, batched_gate in SHAPE_MATRIX:
        results = {name: _result() for name in RESULT_NAMES}
        multipliers = {
            "projected_eager_forward": projected_eager,
            "materialized_pytorch_forward": materialized,
            "compiled_projected_reference_forward": compiled_projected,
            "eager_factorization_forward": eager_factorization,
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
        full_elements = batch * (2 * left_bond) * (2 * right_bond)
        projected_elements = batch * (2 * left_bond) * rank
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "left_bond": left_bond,
                    "physical_dimension": 2,
                    "middle_bond": middle_bond,
                    "right_bond": right_bond,
                    "rank": rank,
                    "batched_gate": batched_gate,
                },
                "contraction_elements": (
                    batch * left_bond * middle_bond * right_bond * rank
                ),
                "full_matrix_elements": full_elements,
                "projected_output_elements": projected_elements,
                "avoided_materialization_ratio": (full_elements / projected_elements),
                "runtime_inference_eligible": True,
                "dtype": "complex64",
                "layout": "contiguous_mps_two_site_projected_range",
                "maximum_sample_absolute_error": 1e-6,
                "maximum_sample_relative_l2_error": 1e-6,
                "factorization_reconstruction_absolute_error": 1e-6,
                "factorization_reconstruction_relative_l2_error": 1e-6,
                "factorization_subspace_absolute_error": 1e-6,
                "factorization_subspace_relative_l2_error": 1e-6,
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
        "rollout_environment": {
            "FQ_TRITON_MPS_PROJECTED_TWO_SITE": "1",
        },
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
            "timing_order": "counterbalanced_forward_reverse_per_repeat",
            "memory_collection": "separate_single_invocation_after_timing",
            "warmup": 2,
            "repeats": 3,
            "group_size": 4,
            "seed": 270002,
            "directions": ["forward"],
        },
        "cases": cases,
    }


def _matrix(
    *,
    projected_eager: float = 1.1,
    materialized: float = 1.2,
    compiled_projected: float = 0.9,
    eager_factorization: float = 1.1,
) -> list[dict[str, object]]:
    return [
        _run(
            host,
            lane,
            projected_eager=projected_eager,
            materialized=materialized,
            compiled_projected=compiled_projected,
            eager_factorization=eager_factorization,
        )
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_forward_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


def test_measurement_counterbalances_order_and_separates_memory(monkeypatch) -> None:
    calls: list[str] = []
    clock = iter(float(index) for index in range(100))
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda: 100)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 140)

    results = _measure_counterbalanced(
        {
            "first": lambda: calls.append("first"),
            "second": lambda: calls.append("second"),
            "third": lambda: calls.append("third"),
        },
        warmup=0,
        repeats=2,
        group_size=1,
    )

    assert calls == [
        "first",
        "second",
        "third",
        "third",
        "second",
        "first",
        "first",
        "second",
        "third",
    ]
    assert all(
        len(result["samples_seconds_per_invocation"]) == 2
        for result in results.values()
    )
    assert all(result["peak_memory_delta_bytes"] == 40 for result in results.values())


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


@pytest.mark.parametrize(
    "field",
    (
        "contraction_elements",
        "full_matrix_elements",
        "projected_output_elements",
        "avoided_materialization_ratio",
    ),
)
def test_run_validator_recomputes_shape_derived_fields(field: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][field] = 1

    with pytest.raises(ValueError, match=f"{field} is not reproducible"):
        validate_run(payload)


def test_run_validator_requires_inference_eligibility() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["runtime_inference_eligible"] = False

    with pytest.raises(ValueError, match="runtime_inference_eligible"):
        validate_run(payload)


def test_run_validator_requires_forward_only_direction() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["measurement"]["directions"] = ["forward", "backward"]

    with pytest.raises(ValueError, match="forward only"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_allows_default_only_after_end_to_end_wins() -> None:
    payload = merge_runs(
        _matrix(projected_eager=1.1, eager_factorization=1.2),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["catalog_win_over_projected_eager_on_all_cases"]
    assert payload["public_factorization_win_over_eager_on_all_cases"]
    assert payload["projected_kernel_dispatch_decision"] == "eligible_for_default"
    assert payload["projected_kernel_dispatch_blockers"] == []
    assert payload["fixed_rank_rollout_decision"] == "retain_opt_in"


@pytest.mark.parametrize(
    ("projected_eager", "eager_factorization"),
    ((0.9, 1.2), (1.1, 0.9)),
)
def test_aggregate_retains_opt_in_when_a_required_comparison_loses(
    projected_eager: float,
    eager_factorization: float,
) -> None:
    payload = merge_runs(
        _matrix(
            projected_eager=projected_eager,
            eager_factorization=eager_factorization,
        ),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["projected_kernel_dispatch_decision"] == "retain_opt_in"
    assert payload["projected_kernel_dispatch_blockers"]
    assert payload["fixed_rank_rollout_decision"] == "retain_opt_in"


def test_aggregate_validation_rejects_noncanonical_decision() -> None:
    payload = merge_runs(_matrix(), required_hosts=("jp-a800-171", "jp-a800-172"))
    changed = copy.deepcopy(payload)
    changed["projected_kernel_dispatch_decision"] = "retain_opt_in"

    with pytest.raises(ValueError, match="canonical merge"):
        validate_evidence(changed)


def test_checked_in_a800_evidence_is_canonical_and_retains_opt_in() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert not payload["direct_win_over_projected_eager_on_all_cases"]
    assert not payload["direct_win_over_materialized_pytorch_on_all_cases"]
    assert not payload["catalog_win_over_projected_eager_on_all_cases"]
    assert not payload["catalog_win_over_compiled_projected_on_all_cases"]
    assert not payload["public_factorization_win_over_eager_on_all_cases"]
    assert payload["direct_memory_win_over_projected_eager_on_all_cases"]
    assert payload["direct_memory_win_over_materialized_pytorch_on_all_cases"]
    assert payload["projected_kernel_dispatch_decision"] == "retain_opt_in"
    assert payload["fixed_rank_rollout_decision"] == "retain_opt_in"
