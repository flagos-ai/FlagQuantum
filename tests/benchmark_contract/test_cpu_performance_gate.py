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
