from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"
REPORT = RESULTS / "LINUX_X86_GCC11_NATIVE_CPU_ADJOINT_20261003.md"
ARTIFACTS = (
    "linux_x86_gcc11_adjoint_18q_1t_20261003.json",
    "linux_x86_gcc11_adjoint_22q_1t_20261003.json",
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_linux_gcc11_report_links_measured_artifacts() -> None:
    document = REPORT.read_text(encoding="utf-8")

    for name in ARTIFACTS:
        assert (RESULTS / name).is_file()
        assert f"]({name})" in document


def test_linux_gcc11_headlines_match_raw_measurements() -> None:
    document = REPORT.read_text(encoding="utf-8")
    expected = {
        (18, "hardware_efficient_vqe"): (42.524680495262146, 190.45613147318363),
        (18, "qaoa_path_maxcut"): (49.2219515144825, 128.45994159579277),
        (22, "hardware_efficient_vqe"): (760.9668355435133, 4425.672495737672),
        (22, "qaoa_path_maxcut"): (942.467950284481, 2800.8579034358263),
    }

    observed: dict[tuple[int, str], tuple[float, float]] = {}
    for name in ARTIFACTS:
        payload = _load(name)
        assert payload["passed"] is True
        assert payload["correctness_passed"] is True
        assert payload["all_measurements_stable"] is True
        assert payload["environment"]["machine"] == "x86_64"
        assert payload["platform"].startswith("Linux-")
        assert payload["environment"]["torch_threads"] == 1

        for case in payload["cases"]:
            workload = case["workload"]
            engines = case["engines"]
            fq = engines["flagquantum_adjoint"]["value_and_grad"]
            lightning = engines["pennylane_lightning_adjoint"]["value_and_grad"]
            key = (int(workload["n_wires"]), str(workload["name"]))
            observed[key] = (
                float(fq["median_seconds"]) * 1000,
                float(lightning["median_seconds"]) * 1000,
            )
            assert fq["sample_count"] == 7
            assert lightning["sample_count"] == 7
            assert case["correctness"]["passed"] is True
            assert case["stability"]["passed"] is True

    assert observed == pytest.approx(expected)
    assert "**4.479x**" in document
    assert "**2.610x**" in document
    assert "**5.816x**" in document
    assert "**2.972x**" in document


def test_linux_gcc11_report_keeps_claim_boundaries() -> None:
    document = REPORT.read_text(encoding="utf-8")

    assert "one pinned socket-local CPU run" in document
    assert "not an additional timing sample" in document
    assert "does not establish multi-socket scaling" in document
    assert "not" in document and "universal framework" in document
    assert "A future\n  Linux optimization" in document
