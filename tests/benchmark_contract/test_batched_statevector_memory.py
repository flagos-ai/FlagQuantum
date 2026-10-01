from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.batched_statevector_memory import (
    SCHEMA,
    render_markdown,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_memory_runner_is_registered_and_lazy() -> None:
    assert "batched_statevector_memory" in runners.names()
    assert callable(runners.resolve("batched_statevector_memory"))


@pytest.mark.skipif(not Path("/usr").exists(), reason="requires a Unix process model")
def test_small_isolated_memory_probe_records_time_rss_and_correctness() -> None:
    payload = run_benchmark(
        workloads=("hardware_efficient_statevector",),
        n_wires=(4,),
        batch_sizes=(3,),
        engines=("flagquantum_native_batch", "flagquantum_native_monolithic_batch"),
        threads=1,
        warmup=0,
        iterations=3,
        memory_probes=2,
    )

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert (
        payload["distribution_semantics"] == "median_of_fresh_processes_per_engine_case"
    )
    case = payload["cases"][0]
    for result in case["engines"].values():
        memory = result["isolated_memory"]
        assert memory["measurement"] == "fresh_process_ru_maxrss"
        assert memory["aggregation"] == "median"
        assert memory["sample_count"] == 2
        assert len(memory["samples"]) == 2
        assert memory["peak_rss_bytes"] > 0
        assert memory["peak_rss_bytes"] >= memory["pre_execution_peak_rss_bytes"]
        assert memory["cold_execution_seconds"] > 0
        assert memory["output_shape"] == [3, 16]
        assert memory["maximum_norm_error"] <= 1e-10

    report = render_markdown(payload, artifact_name="result.json")
    assert "PennyLane" not in report
    assert "Peak RSS (MiB)" in report
    assert "fresh process" in report


def test_checked_in_memory_artifact_is_complete_and_stable() -> None:
    path = (
        REPOSITORY_ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "batched_statevector_memory_cpu_arm64_20261001.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert len(payload["cases"]) == 5
    for case in payload["cases"]:
        assert len(case["engines"]) == 6
        for result in case["engines"].values():
            assert result["batch_total"]["sample_count"] == 11
            assert result["batch_total"]["relative_median_absolute_deviation"] <= 0.20
            assert result["isolated_memory"]["peak_rss_bytes"] > 0
            assert result["isolated_memory"]["maximum_norm_error"] <= 1e-10


def test_checked_in_preallocation_artifact_uses_robust_memory_samples() -> None:
    path = (
        REPOSITORY_ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "batched_statevector_memory_preallocation_cpu_arm64_20261001.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert len(payload["cases"]) == 5
    for case in payload["cases"]:
        assert set(case["engines"]) == {
            "flagquantum_native_batch",
            "flagquantum_native_functional_windows",
        }
        optimized = case["engines"]["flagquantum_native_batch"]
        legacy = case["engines"]["flagquantum_native_functional_windows"]
        assert optimized["isolated_memory"]["sample_count"] == 3
        assert len(optimized["isolated_memory"]["samples"]) == 3
        assert (
            optimized["isolated_memory"]["peak_rss_bytes"]
            < legacy["isolated_memory"]["peak_rss_bytes"]
        )
        assert optimized["batch_total"]["relative_median_absolute_deviation"] <= 0.20
        assert legacy["batch_total"]["relative_median_absolute_deviation"] <= 0.20


def test_checked_in_layout_lifetime_artifact_is_correct_stable_and_lower_rss() -> None:
    path = (
        REPOSITORY_ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "batched_statevector_layout_lifetime_cpu_arm64_20261001.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert set(case["engines"]) == {
            "flagquantum_native_batch",
            "flagquantum_native_layout_retention",
        }
        optimized = case["engines"]["flagquantum_native_batch"]
        retained = case["engines"]["flagquantum_native_layout_retention"]
        assert optimized["isolated_memory"]["sample_count"] == 3
        assert len(optimized["isolated_memory"]["samples"]) == 3
        assert (
            optimized["isolated_memory"]["peak_rss_bytes"]
            < retained["isolated_memory"]["peak_rss_bytes"]
        )
        assert optimized["batch_total"]["relative_median_absolute_deviation"] <= 0.20
        assert retained["batch_total"]["relative_median_absolute_deviation"] <= 0.20
