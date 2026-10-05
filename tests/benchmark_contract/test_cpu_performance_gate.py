from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.cpu_performance_gate import evaluate

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"


def _payload(*, schema: str = "flagquantum.simulator_workload_corpus.v1") -> dict:
    metric_names = (
        ("end_to_end",)
        if schema == "flagquantum.simulator_workload_corpus.v1"
        else ("forward", "backward", "value_and_grad")
    )
    metrics = {
        name: {"median_seconds": 1.0, "sample_count": 7} for name in metric_names
    }
    engine_name = (
        "flagquantum_native"
        if schema == "flagquantum.simulator_workload_corpus.v1"
        else "flagquantum_adjoint"
    )
    return {
        "schema": schema,
        "platform": "testOS-arm64",
        "python": "3.12.7",
        "environment": {
            "machine": "arm64",
            "device": "cpu",
            "torch": "2.13.1",
            "torch_threads": 1,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        },
        "methodology": {
            "measurement_scope": "one_run",
            "calls_per_sample": 1,
            "exact_statevector": True,
        },
        "cases": [
            {
                "workload": {
                    "name": "representative",
                    "n_wires": 18,
                    "dtype": "complex128",
                    "ir_content_hash": "abc123",
                },
                "correctness": {"passed": True},
                "stability": {"passed": True, "engines": {engine_name: True}},
                "engines": {engine_name: metrics},
            }
        ],
    }


def _batched_payload() -> dict:
    return {
        "schema": "flagquantum.batched_statevector_memory.v1",
        "platform": "testOS-arm64",
        "python": "3.12.7",
        "environment": {
            "machine": "arm64",
            "device": "cpu",
            "torch": "2.13.1",
            "torch_threads": 1,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        },
        "methodology": {
            "timing_scope": "complete_N_statevector_user_task",
            "memory_scope": "fresh_process_high_water_resident_set",
            "memory_api": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "exact_statevector": True,
        },
        "cases": [
            {
                "workload": {
                    "name": "random_clifford_statevector",
                    "n_wires": 18,
                    "batch_size": 32,
                    "dtype": "complex128",
                    "scalar_ir_content_hash": "batch123",
                },
                "correctness": {
                    "passed": True,
                    "absolute_tolerance": 1e-10,
                    "engines": {
                        "flagquantum_native_batch": {"passed": True},
                        "flagquantum_native_layout_retention": {"passed": True},
                    },
                },
                "stability": {
                    "passed": True,
                    "engines": {"flagquantum_native_batch": True},
                },
                "engines": {
                    "flagquantum_native_batch": {
                        "batch_total": {"median_seconds": 1.0, "sample_count": 11},
                        "isolated_memory": {
                            "peak_rss_bytes": 1_000_000_000,
                            "sample_count": 3,
                        },
                    }
                },
            }
        ],
    }


def _simulator_framework_payload() -> dict:
    payload = _payload()
    case = payload["cases"][0]
    case["correctness"]["engines"] = {
        "flagquantum_native": {"passed": True},
        "qiskit_aer": {"passed": True},
    }
    case["stability"]["engines"]["qiskit_aer"] = True
    case["engines"]["flagquantum_native"]["versions"] = {"flagquantum": "0.2.0"}
    case["engines"]["qiskit_aer"] = {
        "versions": {"qiskit": "2.5.1", "qiskit-aer": "0.17.2"},
        "end_to_end": {"median_seconds": 1.25, "sample_count": 7},
    }
    return payload


def test_runner_is_registered() -> None:
    assert "cpu_performance_gate" in runners.names()
    assert callable(runners.resolve("cpu_performance_gate"))


@pytest.mark.parametrize(
    "schema",
    [
        "flagquantum.simulator_workload_corpus.v1",
        "flagquantum.differentiable_simulator_corpus.v1",
    ],
)
def test_gate_accepts_supported_corpora_within_tolerance(schema: str) -> None:
    baseline = _payload(schema=schema)
    current = copy.deepcopy(baseline)
    engine = next(iter(current["cases"][0]["engines"].values()))
    for metric in engine.values():
        metric["median_seconds"] = 1.19

    report = evaluate(baseline, current)

    assert report["verdict"] == "pass"
    assert report["passed"] is True
    assert report["cases"][0]["passed"] is True


def test_gate_accepts_batched_time_and_peak_rss_within_tolerance() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    native = current["cases"][0]["engines"]["flagquantum_native_batch"]
    native["batch_total"]["median_seconds"] = 1.19
    native["isolated_memory"]["peak_rss_bytes"] = 1_090_000_000

    report = evaluate(baseline, current)

    assert report["verdict"] == "pass"
    metrics = report["cases"][0]["metrics"]
    assert metrics["batch_total"]["current_over_baseline"] == 1.19
    assert metrics["peak_rss"]["current_over_baseline"] == 1.09
    assert report["policy"]["minimum_memory_probes"] == 3
    assert (
        report["distribution_semantics"]
        == "timing_single_process_memory_median_of_fresh_processes"
    )


def test_batched_gate_fails_closed_on_memory_growth_and_probe_count() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    memory = current["cases"][0]["engines"]["flagquantum_native_batch"][
        "isolated_memory"
    ]
    memory["peak_rss_bytes"] = 1_110_000_000
    memory["sample_count"] = 2

    report = evaluate(baseline, current)

    assert report["verdict"] == "fail"
    failures = "\n".join(report["failures"])
    assert "peak_rss has 2 probes; requires 3" in failures
    assert "peak_rss growth 1.110 exceeds 1.100" in failures


def test_batched_gate_rejects_self_referenced_correctness() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    current["cases"][0]["correctness"]["engines"] = {
        "flagquantum_native_batch": {"passed": True}
    }

    report = evaluate(baseline, current)

    assert report["verdict"] == "fail"
    assert "independent correctness comparison missing" in report["failures"][0]


def test_batched_gate_rejects_failed_reference_or_looser_tolerance() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    current["cases"][0]["correctness"]["absolute_tolerance"] = 1e-8
    current["cases"][0]["correctness"]["engines"][
        "flagquantum_native_layout_retention"
    ]["passed"] = False

    report = evaluate(baseline, current)

    failures = "\n".join(report["failures"])
    assert "correctness engine failed" in failures
    assert "absolute tolerance changed from 1e-10 to 1e-08" in failures


def test_batched_gate_treats_memory_method_changes_as_incomparable() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    current["methodology"]["memory_scope"] = "allocator_only"

    report = evaluate(baseline, current)

    assert report["verdict"] == "incomparable"
    assert "memory_scope" in report["profile_differences"]


def test_batched_gate_can_lock_a_framework_comparison() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    case = current["cases"][0]
    case["engines"]["pennylane_lightning_bridge"] = {
        "batch_total": {"median_seconds": 1.25, "sample_count": 11}
    }
    case["correctness"]["engines"]["pennylane_lightning_bridge"] = {"passed": True}
    case["stability"]["engines"]["pennylane_lightning_bridge"] = True

    report = evaluate(
        baseline,
        current,
        comparison_engine="pennylane_lightning_bridge",
        max_native_over_comparison=1.0,
    )

    comparison = report["cases"][0]["metrics"]["comparison"]
    assert report["verdict"] == "pass"
    assert comparison["native_over_comparison"] == 0.8
    assert comparison["passed"] is True


def test_batched_framework_gate_fails_closed() -> None:
    baseline = _batched_payload()
    current = copy.deepcopy(baseline)
    case = current["cases"][0]
    case["engines"]["pennylane_lightning_bridge"] = {
        "batch_total": {"median_seconds": 0.8, "sample_count": 4}
    }
    case["correctness"]["engines"]["pennylane_lightning_bridge"] = {"passed": True}
    case["stability"]["engines"]["pennylane_lightning_bridge"] = False

    report = evaluate(
        baseline,
        current,
        comparison_engine="pennylane_lightning_bridge",
    )

    failures = "\n".join(report["failures"])
    assert report["verdict"] == "fail"
    assert "has 4 samples; requires 5" in failures
    assert "comparison measurement stability failed" in failures
    assert "native/comparison ratio 1.250 exceeds 1.000" in failures


def test_simulator_gate_can_lock_an_external_framework_floor() -> None:
    baseline = _simulator_framework_payload()
    current = copy.deepcopy(baseline)

    report = evaluate(
        baseline,
        current,
        comparison_engine="qiskit_aer",
        max_native_over_comparison=0.90,
    )

    comparison = report["cases"][0]["metrics"]["comparison"]
    assert report["verdict"] == "pass"
    assert comparison["native_over_comparison"] == 0.8
    assert comparison["passed"] is True


def test_simulator_framework_gate_fails_closed_on_version_and_speed() -> None:
    baseline = _simulator_framework_payload()
    current = copy.deepcopy(baseline)
    current_case = current["cases"][0]
    current_case["engines"]["qiskit_aer"]["versions"]["qiskit-aer"] = "0.18.0"
    current_case["engines"]["qiskit_aer"]["end_to_end"]["median_seconds"] = 0.8

    report = evaluate(
        baseline,
        current,
        comparison_engine="qiskit_aer",
    )

    failures = "\n".join(report["failures"])
    assert report["verdict"] == "fail"
    assert "comparison engine versions changed: qiskit_aer" in failures
    assert "native/comparison ratio 1.250 exceeds 1.000" in failures


def test_framework_gate_rejects_missing_or_inapplicable_comparison() -> None:
    report = evaluate(
        _batched_payload(),
        _batched_payload(),
        comparison_engine="pennylane_lightning_bridge",
    )
    assert report["verdict"] == "fail"
    assert "comparison engine missing" in report["failures"][0]

    with pytest.raises(ValueError, match="only for statevector simulator corpora"):
        evaluate(
            _payload(schema="flagquantum.differentiable_simulator_corpus.v1"),
            _payload(schema="flagquantum.differentiable_simulator_corpus.v1"),
            comparison_engine="external",
        )


def test_gate_fails_closed_on_regression_correctness_stability_and_samples() -> None:
    baseline = _payload()
    current = copy.deepcopy(baseline)
    case = current["cases"][0]
    case["correctness"]["passed"] = False
    case["stability"]["passed"] = False
    case["stability"]["engines"]["flagquantum_native"] = False
    metric = case["engines"]["flagquantum_native"]["end_to_end"]
    metric["median_seconds"] = 1.21
    metric["sample_count"] = 4

    report = evaluate(baseline, current)

    assert report["verdict"] == "fail"
    assert report["passed"] is False
    failures = "\n".join(report["failures"])
    assert "correctness failed" in failures
    assert "native measurement stability failed" in failures
    assert "slowdown 1.210 exceeds 1.200" in failures
    assert "has 4 samples; requires 5" in failures


def test_gate_reports_incomparable_profile_instead_of_false_verdict() -> None:
    baseline = _payload()
    current = copy.deepcopy(baseline)
    current["environment"]["torch_threads"] = 2

    report = evaluate(baseline, current)

    assert report["verdict"] == "incomparable"
    assert report["passed"] is False
    assert report["profile_differences"]["torch_threads"] == {
        "baseline": 1,
        "current": 2,
    }


def test_gate_requires_matching_cpu_model_and_affinity_when_recorded() -> None:
    baseline = _payload()
    current = copy.deepcopy(baseline)
    baseline["environment"].update({"cpu_model": "Example CPU", "cpu_affinity": [0, 1]})
    current["environment"].update({"cpu_model": "Other CPU", "cpu_affinity": [2, 3]})

    report = evaluate(baseline, current)

    assert report["verdict"] == "incomparable"
    assert set(report["profile_differences"]) == {"cpu_model", "cpu_affinity"}


def test_gate_fails_when_baseline_case_disappears() -> None:
    baseline = _payload()
    current = copy.deepcopy(baseline)
    current["cases"] = []

    report = evaluate(baseline, current)

    assert report["verdict"] == "fail"
    assert report["missing_cases"]
    assert report["failures"][0].startswith("missing case:")


def test_gate_rejects_workload_identity_or_schema_ambiguity() -> None:
    baseline = _payload()
    duplicate = copy.deepcopy(baseline["cases"][0])
    baseline["cases"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate benchmark case"):
        evaluate(baseline, _payload())

    with pytest.raises(ValueError, match="same supported schema"):
        evaluate(
            _payload(),
            _payload(schema="flagquantum.differentiable_simulator_corpus.v1"),
        )


def test_gate_rejects_missing_profile_and_invalid_timings() -> None:
    missing_profile = _payload()
    del missing_profile["environment"]["torch"]
    with pytest.raises(ValueError, match="environment.torch"):
        evaluate(missing_profile, _payload())

    invalid_timing = _payload()
    invalid_timing["cases"][0]["engines"]["flagquantum_native"]["end_to_end"][
        "median_seconds"
    ] = float("nan")
    with pytest.raises(ValueError, match="current median must be positive"):
        evaluate(_payload(), invalid_timing)


@pytest.mark.parametrize(
    ("baseline_name", "current_name", "report_name", "case_count"),
    [
        (
            "simulator_workload_corpus_cpu_arm64_20260924.json",
            "cpu_phase1_forward_cpu_arm64_20260930.json",
            "cpu_phase1_forward_gate_cpu_arm64_20260930.json",
            20,
        ),
        (
            "adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json",
            "cpu_phase1_adjoint_cpu_arm64_20260930.json",
            "cpu_phase1_adjoint_gate_cpu_arm64_20260930.json",
            8,
        ),
        (
            "batched_statevector_layout_lifetime_cpu_arm64_20261001.json",
            "batched_statevector_regression_current_cpu_arm64_20261001.json",
            "batched_statevector_regression_gate_cpu_arm64_20261001.json",
            2,
        ),
    ],
)
def test_checked_in_cpu_scorecard_is_reproducible(
    baseline_name: str, current_name: str, report_name: str, case_count: int
) -> None:
    def load(name: str) -> dict:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    expected = load(report_name)
    regenerated = evaluate(load(baseline_name), load(current_name))

    assert expected == regenerated
    assert regenerated["verdict"] == "pass"
    assert len(regenerated["cases"]) == case_count


def test_checked_in_framework_floor_is_reproducible() -> None:
    baseline = json.loads(
        (
            RESULTS
            / "batched_statevector_framework_gate_baseline_cpu_arm64_20261002.json"
        ).read_text(encoding="utf-8")
    )
    expected = json.loads(
        (
            RESULTS / "batched_statevector_framework_gate_cpu_arm64_20261002.json"
        ).read_text(encoding="utf-8")
    )

    regenerated = evaluate(
        baseline,
        baseline,
        max_slowdown=1.10,
        max_memory_growth=1.10,
        minimum_samples=5,
        minimum_memory_probes=3,
        comparison_engine="pennylane_lightning_bridge",
        max_native_over_comparison=0.90,
    )

    assert regenerated == expected
    assert regenerated["verdict"] == "pass"
    assert len(regenerated["cases"]) == 2
    assert all(
        case["metrics"]["comparison"]["native_over_comparison"] <= 0.90
        for case in regenerated["cases"]
    )


@pytest.mark.parametrize(
    ("engine", "report_name"),
    [
        ("qiskit_aer", "linux_x86_cpu_regression_gate_20261004.json"),
        (
            "pennylane_lightning_qubit",
            "linux_x86_cpu_regression_gate_pennylane_20261004.json",
        ),
    ],
)
def test_checked_in_linux_x86_framework_floor_is_reproducible(
    engine: str,
    report_name: str,
) -> None:
    baseline = json.loads(
        (RESULTS / "linux_x86_cpu_regression_baseline_20261004.json").read_text(
            encoding="utf-8"
        )
    )
    expected = json.loads((RESULTS / report_name).read_text(encoding="utf-8"))

    regenerated = evaluate(
        baseline,
        baseline,
        max_slowdown=1.20,
        minimum_samples=7,
        comparison_engine=engine,
        max_native_over_comparison=1.0,
    )

    assert regenerated == expected
    assert regenerated["verdict"] == "pass"
