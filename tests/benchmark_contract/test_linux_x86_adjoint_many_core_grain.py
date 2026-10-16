from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"
REPORT = RESULTS / "NATIVE_CPU_ADJOINT_MANY_CORE_GRAIN_LINUX_X86_20261003.md"
ARTIFACTS = (
    "linux_x86_adjoint_grain_22q_32t_20261003.json",
    "linux_x86_adjoint_grain_22q_32t_node_b_20261003.json",
    "linux_x86_adjoint_grain_22q_32t_split_numa_20261003.json",
    "linux_x86_adjoint_grain_22q_64t_dual_socket_20261003.json",
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _total_ms(case: dict[str, Any], engine: str) -> float:
    return float(case["engines"][engine]["value_and_grad"]["median_seconds"]) * 1000


def test_many_core_report_links_every_raw_artifact() -> None:
    document = REPORT.read_text(encoding="utf-8")

    for name in ARTIFACTS:
        assert (RESULTS / name).is_file()
        assert f"]({name})" in document


def test_primary_many_core_headlines_match_raw_measurements() -> None:
    payload = _load(ARTIFACTS[0])
    document = REPORT.read_text(encoding="utf-8")

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["environment"]["machine"] == "x86_64"
    assert payload["environment"]["torch_threads"] == 32

    expected = {
        "hardware_efficient_vqe": (64.54450264573097, 78.06260697543621),
        "qaoa_path_maxcut": (61.511049047112465, 82.85968378186226),
    }
    observed: dict[str, tuple[float, float]] = {}
    for case in payload["cases"]:
        name = str(case["workload"]["name"])
        observed[name] = (
            _total_ms(case, "flagquantum_adjoint"),
            _total_ms(case, "flagquantum_adjoint_parallel_grain_rollback"),
        )
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        for engine in case["engines"].values():
            assert engine["value_and_grad"]["sample_count"] == 7

    assert observed == pytest.approx(expected)
    assert "**1.209x**" in document
    assert "**1.347x**" in document
    assert "**55.511x**" in document
    assert "**45.142x**" in document


def test_independent_node_reproduces_the_improvement() -> None:
    payload = _load(ARTIFACTS[1])

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    for case in payload["cases"]:
        optimized = _total_ms(case, "flagquantum_adjoint")
        rollback = _total_ms(case, "flagquantum_adjoint_parallel_grain_rollback")
        assert rollback / optimized >= 1.18


def test_numa_artifacts_and_report_keep_claim_boundaries() -> None:
    split = _load(ARTIFACTS[2])
    dual = _load(ARTIFACTS[3])
    document = REPORT.read_text(encoding="utf-8")

    assert split["environment"]["torch_threads"] == 32
    assert dual["environment"]["torch_threads"] == 64
    assert split["passed"] is True and split["all_measurements_stable"] is True
    assert dual["passed"] is True and dual["all_measurements_stable"] is True
    assert "does not introduce a NUMA policy" in document
    assert "not a universal framework\nranking" in document
    assert "does not establish universal thread scaling" in document
