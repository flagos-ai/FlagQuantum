from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"
REPORT = RESULTS / "LINUX_X86_CPU_SCORECARD_20261003.md"
ARTIFACTS = {
    1: "linux_x86_cpu_scorecard_forward_1t_20261003.json",
    32: "linux_x86_cpu_scorecard_forward_32t_20261003.json",
}
WORKLOADS = {
    "hardware_efficient_statevector",
    "truncated_qft_statevector",
    "random_clifford_statevector",
    "local_brickwork_statevector",
    "dense_nonlocal_statevector",
    "swap_routing_statevector",
}
ENGINES = {
    "flagquantum_native",
    "qiskit_aer",
    "pennylane_lightning_qubit",
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _cases(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(case["workload"]["name"]), int(case["workload"]["n_wires"])): case
        for case in payload["cases"]
    }


@pytest.mark.parametrize(("threads", "name"), ARTIFACTS.items())
def test_linux_x86_forward_artifacts_are_complete_and_stable(
    threads: int, name: str
) -> None:
    payload = _load(name)

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["environment"]["machine"] == "x86_64"
    assert payload["environment"]["torch_threads"] == threads
    assert set(payload["engines"]) == ENGINES
    assert set(payload["workloads"]) == WORKLOADS
    assert payload["n_wires"] == [18, 22]
    assert len(payload["cases"]) == 12

    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert set(case["engines"]) == ENGINES
        for engine in case["engines"].values():
            assert engine["end_to_end"]["sample_count"] == 7


def test_scorecard_headlines_are_derived_from_raw_measurements() -> None:
    one = _cases(_load(ARTIFACTS[1]))
    socket = _cases(_load(ARTIFACTS[32]))
    document = REPORT.read_text(encoding="utf-8")

    for key, one_case in one.items():
        socket_case = socket[key]
        for case in (one_case, socket_case):
            for engine in ENGINES:
                milliseconds = (
                    case["engines"][engine]["end_to_end"]["median_seconds"] * 1000
                )
                assert f"{milliseconds:,.3f} ms" in document

            native = case["engines"]["flagquantum_native"]["end_to_end"]
            external = min(
                case["engines"][engine]["end_to_end"]["median_seconds"]
                for engine in ("qiskit_aer", "pennylane_lightning_qubit")
            )
            assert f"**{external / native['median_seconds']:.3f}x**" in document

        one_native = one_case["engines"]["flagquantum_native"]["end_to_end"]
        socket_native = socket_case["engines"]["flagquantum_native"]["end_to_end"]
        speedup = one_native["median_seconds"] / socket_native["median_seconds"]
        assert f"**{speedup:.3f}x**" in document

    assert "FlagQuantum gains 1.07x-6.99x, not 32x" in document
    assert "Qiskit Aer remains faster in\nfour of those six workloads" in document


def test_scorecard_preserves_comparison_and_kokkos_boundaries() -> None:
    document = REPORT.read_text(encoding="utf-8")
    index = (RESULTS / "README.md").read_text(encoding="utf-8")

    for name in ARTIFACTS.values():
        assert f"]({name})" in document
    assert "NATIVE_CPU_ADJOINT_MANY_CORE_GRAIN_LINUX_X86_20261003.md" in document
    assert "not available" in document
    assert "not a FlagQuantum dependency" in document
    assert "not be used\n  as an isolated native-kernel ranking" in document
    assert "LINUX_X86_CPU_SCORECARD_20261003.md" in index
