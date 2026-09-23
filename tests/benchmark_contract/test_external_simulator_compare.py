from __future__ import annotations

import json
from pathlib import Path

import pytest

from flagquantum.benchmarking.external_simulator_compare import (
    run_benchmark,
    run_case,
)
from flagquantum.benchmarking.simulator_compare import ABSOLUTE_TOLERANCE, SCHEMA

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]


def test_external_comparison_rejects_an_empty_size_matrix() -> None:
    with pytest.raises(ValueError, match="at least one workload size"):
        run_benchmark(
            engine="cirq_simulator",
            n_wires=(),
            layers=1,
            threads=1,
            warmup=0,
            iterations=3,
            setup_iterations=1,
            calls_per_sample=1,
        )


def test_external_comparison_rejects_too_few_iterations() -> None:
    with pytest.raises(ValueError, match="iterations at least 3"):
        run_case(
            engine="cirq_simulator",
            n_wires=4,
            layers=1,
            threads=1,
            warmup=0,
            iterations=2,
            setup_iterations=1,
            calls_per_sample=1,
        )


def _assert_smoke_payload(payload: dict[str, object], engine: str) -> None:
    assert payload["passed"] is True
    assert payload["schema"] == SCHEMA
    assert payload["measured_engine"] == engine
    methodology = payload["methodology"]
    assert isinstance(methodology, dict)
    assert methodology["measurement_scope"] == "external_engine_only"
    assert methodology["flagquantum_performance_measured"] is False
    cases = payload["cases"]
    assert isinstance(cases, tuple)
    case = cases[0]
    assert set(case["engines"]) == {engine}
    assert case["correctness"]["flagquantum_reference_timed"] is False
    assert case["correctness"]["max_abs_error"] <= ABSOLUTE_TOLERANCE


@pytest.mark.cirq
def test_cirq_external_comparison_smoke_payload() -> None:
    pytest.importorskip("cirq")
    payload = run_benchmark(
        engine="cirq_simulator",
        n_wires=(4,),
        layers=1,
        threads=1,
        warmup=0,
        iterations=3,
        setup_iterations=1,
        calls_per_sample=1,
    )
    _assert_smoke_payload(payload, "cirq_simulator")
    assert payload["runner"] == "simulator_compare_cirq"


@pytest.mark.pennylane
def test_pennylane_external_comparison_smoke_payload() -> None:
    pytest.importorskip("pennylane")
    payload = run_benchmark(
        engine="pennylane_lightning_qubit",
        n_wires=(4,),
        layers=1,
        threads=1,
        warmup=0,
        iterations=3,
        setup_iterations=1,
        calls_per_sample=1,
    )
    _assert_smoke_payload(payload, "pennylane_lightning_qubit")
    assert payload["runner"] == "simulator_compare_pennylane"


@pytest.mark.parametrize(
    ("filename", "engine"),
    (
        ("cirq_cpu_arm64_20260923.json", "cirq_simulator"),
        (
            "pennylane_lightning_cpu_arm64_20260923.json",
            "pennylane_lightning_qubit",
        ),
    ),
)
def test_checked_in_external_measurement_artifact(filename: str, engine: str) -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / filename
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["measured_engine"] == engine
    assert payload["methodology"]["measurement_scope"] == "external_engine_only"
    assert payload["methodology"]["flagquantum_performance_measured"] is False
    assert [case["workload"]["n_wires"] for case in payload["cases"]] == [
        10,
        14,
        18,
        22,
        24,
    ]
    for case in payload["cases"]:
        assert set(case["engines"]) == {engine}
        assert case["correctness"]["passed"] is True
        assert case["correctness"]["flagquantum_reference_timed"] is False
        report = case["engines"][engine]["conversion_report"]
        assert report["lossless"] is True
        assert report["issues"] == []
