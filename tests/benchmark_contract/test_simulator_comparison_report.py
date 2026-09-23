from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flagquantum.benchmarking.simulator_comparison_report import (
    SCHEMA,
    build_report,
    render_markdown,
)

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"
SOURCES = (
    RESULTS / "flagquantum_qiskit_aer_cpu_arm64_20260923.json",
    RESULTS / "cirq_cpu_arm64_20260923.json",
    RESULTS / "pennylane_lightning_cpu_arm64_20260923.json",
)
REPORT_JSON = RESULTS / "simulator_comparison_cpu_arm64_20260923.json"
REPORT_MARKDOWN = RESULTS / "SIMULATOR_COMPARISON_CPU_ARM64_20260923.md"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_combines_measured_times_ratios_and_stability() -> None:
    report = build_report(SOURCES)

    assert report["schema"] == SCHEMA
    assert report["passed"] is True
    assert report["all_measurements_stable"] is False
    assert report["engine_order"] == [
        "flagquantum_native",
        "qiskit_aer",
        "cirq_simulator",
        "pennylane_lightning_qubit",
    ]
    first = report["rows"][0]
    assert first["workload"]["n_wires"] == 10
    assert first["engines"]["flagquantum_native"][
        "median_milliseconds"
    ] == pytest.approx(0.5772208038251847)
    assert first["ratios_to_flagquantum"]["qiskit_aer"] == pytest.approx(
        1.1576592465431044
    )
    assert first["engines"]["cirq_simulator"]["stable"] is False
    assert report["historical_comparison"] == {
        "baseline_supplied": False,
        "regression_gate_enabled": False,
    }


def test_report_rejects_incompatible_workload_matrix(tmp_path: Path) -> None:
    cirq = _read(SOURCES[1])
    cases = cirq["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    workload = first["workload"]
    assert isinstance(workload, dict)
    workload["gate_count"] = 83
    changed = _write(tmp_path / "cirq.json", cirq)

    with pytest.raises(ValueError, match="incompatible workload matrix"):
        build_report((SOURCES[0], changed))


def test_report_rejects_incompatible_measurement_environment(tmp_path: Path) -> None:
    cirq = _read(SOURCES[1])
    environment = cirq["environment"]
    assert isinstance(environment, dict)
    environment["torch_threads"] = 2
    changed = _write(tmp_path / "cirq.json", cirq)

    with pytest.raises(ValueError, match="incompatible environment or methodology"):
        build_report((SOURCES[0], changed))


def test_historical_comparison_reports_but_does_not_gate_regression(
    tmp_path: Path,
) -> None:
    baseline = build_report(SOURCES)
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    native = copy.deepcopy(_read(SOURCES[0]))
    cases = native["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    engines = first["engines"]
    assert isinstance(engines, dict)
    flagquantum = engines["flagquantum_native"]
    assert isinstance(flagquantum, dict)
    steady_state = flagquantum["steady_state"]
    assert isinstance(steady_state, dict)
    median = steady_state["median_seconds"]
    assert isinstance(median, float)
    steady_state["median_seconds"] = median * 1.2
    current_native = _write(tmp_path / "native.json", native)

    report = build_report(
        (current_native, SOURCES[1], SOURCES[2]),
        baseline=baseline_path,
        regression_threshold_percent=10.0,
    )

    history = report["historical_comparison"]
    assert history["regression_gate_enabled"] is False
    assert history["regression_count"] == 1
    changed = [
        item
        for item in history["changes"]
        if item["n_wires"] == 10 and item["engine"] == "flagquantum_native"
    ]
    assert changed == [
        {
            "workload_fingerprint": report["rows"][0]["workload_fingerprint"],
            "n_wires": 10,
            "engine": "flagquantum_native",
            "baseline_seconds": median,
            "current_seconds": pytest.approx(median * 1.2),
            "change_percent": pytest.approx(20.0),
            "verdict": "regression",
        }
    ]


def test_checked_in_report_is_generated_from_raw_artifacts() -> None:
    report = build_report(tuple(path.relative_to(ROOT) for path in SOURCES))
    expected_json = json.dumps(report, indent=2, sort_keys=True) + "\n"

    assert REPORT_JSON.read_text(encoding="utf-8") == expected_json
    assert REPORT_MARKDOWN.read_text(encoding="utf-8") == render_markdown(report)
