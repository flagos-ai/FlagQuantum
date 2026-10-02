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
    assert (
        payload["methodology"]["pennylane_native_batch_preprocessing_in_timing"]
        is False
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


def test_checked_in_dense_width_artifact_records_speed_memory_and_framework() -> None:
    path = (
        REPOSITORY_ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "batched_statevector_dense_width_cpu_arm64_20261002.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["hostname"] == "redacted"
    assert len(payload["cases"]) == 1
    engines = payload["cases"][0]["engines"]
    assert set(engines) == {
        "flagquantum_native_batch",
        "flagquantum_native_dense_width_rollback",
        "pennylane_lightning_bridge",
    }
    optimized = engines["flagquantum_native_batch"]
    rollback = engines["flagquantum_native_dense_width_rollback"]
    lightning = engines["pennylane_lightning_bridge"]
    assert optimized["batch_total"]["sample_count"] == 11
    assert optimized["isolated_memory"]["sample_count"] == 3
    assert (
        optimized["batch_total"]["median_seconds"]
        < rollback["batch_total"]["median_seconds"]
    )
    assert (
        optimized["batch_total"]["median_seconds"]
        < lightning["batch_total"]["median_seconds"]
    )
    assert (
        optimized["isolated_memory"]["peak_rss_bytes"]
        < rollback["isolated_memory"]["peak_rss_bytes"]
    )


def test_checked_in_pennylane_native_batch_artifact_is_fair_and_complete() -> None:
    comparison = REPOSITORY_ROOT / "benchmarks" / "results" / "comparison"
    path = (
        comparison
        / "batched_statevector_pennylane_native_batch_cpu_arm64_20261002.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = json.loads(
        (
            comparison
            / "batched_statevector_pennylane_native_batch_gate_cpu_arm64_20261002.json"
        ).read_text(encoding="utf-8")
    )
    generated = (
        comparison / "BATCHED_STATEVECTOR_PENNYLANE_NATIVE_BATCH_CPU_ARM64_20261002.md"
    )
    scorecard = (
        comparison
        / "BATCHED_STATEVECTOR_PENNYLANE_NATIVE_BATCH_CPU_ARM64_20261002_SCORECARD.md"
    )

    assert payload["hostname"] == "redacted"
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert (
        payload["methodology"]["pennylane_native_batch_preprocessing_in_timing"]
        is False
    )
    assert len(payload["cases"]) == 5
    assert gate["verdict"] == "pass"
    assert gate["policy"]["comparison_engine"] == "pennylane_lightning_native_batch"
    assert gate["policy"]["max_native_over_comparison"] == 0.90
    assert len(gate["cases"]) == 5
    assert all(case["passed"] for case in gate["cases"])
    for case in payload["cases"]:
        assert set(case["engines"]) == {
            "flagquantum_native_batch",
            "pennylane_lightning_native_batch",
            "pennylane_lightning_bridge",
        }
        native = case["engines"]["flagquantum_native_batch"]
        lightning = case["engines"]["pennylane_lightning_native_batch"]
        assert native["batch_total"]["sample_count"] == 11
        assert lightning["batch_total"]["sample_count"] == 11
        assert native["isolated_memory"]["sample_count"] == 3
        assert lightning["isolated_memory"]["sample_count"] == 3
        assert (
            native["batch_total"]["median_seconds"]
            < lightning["batch_total"]["median_seconds"]
        )
        assert (
            case["correctness"]["engines"]["pennylane_lightning_native_batch"][
                "max_abs_error"
            ]
            <= 1e-10
        )
    assert generated.read_text(encoding="utf-8") == render_markdown(
        payload, artifact_name=path.name
    )
    scorecard_text = scorecard.read_text(encoding="utf-8")
    for case in payload["cases"]:
        native_ms = (
            case["engines"]["flagquantum_native_batch"]["batch_total"]["median_seconds"]
            * 1000
        )
        lightning_ms = (
            case["engines"]["pennylane_lightning_native_batch"]["batch_total"][
                "median_seconds"
            ]
            * 1000
        )
        speedup = lightning_ms / native_ms
        assert f"{native_ms:.3f} ms" in scorecard_text
        assert f"{lightning_ms:.3f} ms" in scorecard_text
        assert f"{speedup:.3f}x" in scorecard_text
