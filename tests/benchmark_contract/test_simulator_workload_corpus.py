from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.benchmarking.simulator_workload_corpus import (
    FEATURE_SCHEMA,
    SCHEMA,
    WORKLOAD_NAMES,
    _refresh_benchmark,
    _render_markdown,
    build_workload,
    extract_features,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]


def test_feature_extraction_reports_depth_connectivity_and_fingerprint() -> None:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.h(0).cx(0, 1).rz(3, 0.2).cz(1, 3)

    features = extract_features(circuit.to_ir())

    assert features == {
        "schema": FEATURE_SCHEMA,
        "n_wires": 4,
        "dtype": "complex128",
        "batch_size": 1,
        "gate_count": 4,
        "logical_depth": 3,
        "single_qubit_gate_count": 2,
        "two_qubit_gate_count": 2,
        "multi_qubit_gate_count": 0,
        "parameterized_gate_count": 1,
        "gate_density_per_wire": 1.0,
        "opcode_histogram": {"cx": 1, "cz": 1, "h": 1, "rz": 1},
        "two_qubit_mean_wire_span": 1.5,
        "two_qubit_max_wire_span": 2,
        "nearest_neighbor_two_qubit_fraction": 0.5,
        "feature_fingerprint": features["feature_fingerprint"],
    }
    assert len(features["feature_fingerprint"]) == 64


def test_workload_corpus_is_deterministic_and_structurally_distinct() -> None:
    first_hashes = {
        name: build_workload(name, n_wires=6).to_ir().content_hash
        for name in WORKLOAD_NAMES
    }
    second_hashes = {
        name: build_workload(name, n_wires=6).to_ir().content_hash
        for name in WORKLOAD_NAMES
    }

    assert first_hashes == second_hashes
    assert len(set(first_hashes.values())) == len(WORKLOAD_NAMES)


def test_swap_routing_workload_has_declared_structure() -> None:
    ir = build_workload("swap_routing_statevector", n_wires=22).to_ir()

    histogram: dict[str, int] = {}
    for instruction in ir.instructions:
        histogram[instruction.name] = histogram.get(instruction.name, 0) + 1

    assert len(ir.instructions) == 150
    assert histogram == {"cx": 11, "h": 11, "ry": 88, "swap": 40}


def test_native_only_corpus_smoke_records_real_timings() -> None:
    payload = run_benchmark(
        workloads=WORKLOAD_NAMES,
        n_wires=(4,),
        engines=("flagquantum_native",),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
    )

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["methodology"]["measurement_scope"] == (
        "end_to_end_user_facing_run_call"
    )
    assert len(payload["cases"]) == len(WORKLOAD_NAMES)
    for case in payload["cases"]:
        timing = case["engines"]["flagquantum_native"]["end_to_end"]
        assert timing["sample_count"] == 3
        assert timing["median_seconds"] > 0
        assert case["correctness"]["passed"] is True
        assert case["features"]["feature_fingerprint"]


def _comparison_baseline() -> tuple[dict[str, Any], dict[str, Any]]:
    measured = run_benchmark(
        workloads=("random_clifford_statevector",),
        n_wires=(4,),
        engines=("flagquantum_native",),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
    )
    baseline = deepcopy(measured)
    baseline["engines"] = [
        "flagquantum_native",
        "qiskit_aer",
        "cirq_simulator",
    ]
    case = baseline["cases"][0]
    native = case["engines"]["flagquantum_native"]
    native_median = native["end_to_end"]["median_seconds"]
    for multiplier, engine in ((2.0, "qiskit_aer"), (3.0, "cirq_simulator")):
        external = deepcopy(native)
        external["end_to_end"]["median_seconds"] = native_median * multiplier
        case["engines"][engine] = external
        case["correctness"]["engines"][engine] = {
            "passed": True,
            "max_abs_error": 0.0,
        }
        case["stability"]["engines"][engine] = True
    case["comparison"]["engine_over_flagquantum_median"] = {
        "qiskit_aer": 2.0,
        "cirq_simulator": 3.0,
    }
    return baseline, measured


def test_native_refresh_preserves_external_payloads_and_recomputes_ratios() -> None:
    baseline, measured = _comparison_baseline()
    original_external = {
        engine: deepcopy(baseline["cases"][0]["engines"][engine])
        for engine in ("qiskit_aer", "cirq_simulator")
    }
    measured["cases"][0]["engines"]["flagquantum_native"]["end_to_end"][
        "median_seconds"
    ] /= 2

    refreshed = _refresh_benchmark(baseline, measured)
    case = refreshed["cases"][0]

    assert baseline.get("refresh_history") is None
    assert {
        engine: case["engines"][engine] for engine in ("qiskit_aer", "cirq_simulator")
    } == original_external
    assert case["comparison"]["engine_over_flagquantum_median"] == pytest.approx(
        {"qiskit_aer": 4.0, "cirq_simulator": 6.0}
    )
    assert refreshed["refresh_history"][-1]["preserved_engines"] == [
        "qiskit_aer",
        "cirq_simulator",
    ]


def test_native_refresh_rejects_methodology_or_workload_drift() -> None:
    baseline, measured = _comparison_baseline()
    measured["methodology"]["iterations"] = 5
    with pytest.raises(ValueError, match="methodology"):
        _refresh_benchmark(baseline, measured)

    baseline, measured = _comparison_baseline()
    measured["cases"][0]["workload"]["ir_content_hash"] = "changed"
    with pytest.raises(ValueError, match="workload does not match"):
        _refresh_benchmark(baseline, measured)


def test_rendered_corpus_report_discloses_native_only_refresh() -> None:
    baseline, measured = _comparison_baseline()
    refreshed = _refresh_benchmark(baseline, measured)

    markdown = _render_markdown(refreshed, artifact_name="corpus.json")

    assert "remeasured only FlagQuantum native" in markdown
    assert "Qiskit Aer, and Cirq" in markdown
    assert "measurements were preserved unchanged" in markdown
    assert "--refresh-from corpus.json" in markdown
    assert "--workloads random_clifford_statevector" in markdown
    assert "| Random Clifford | 4 |" in markdown

    complete_markdown = _render_markdown(baseline, artifact_name="corpus.json")
    assert "Reproduce the complete cross-framework corpus" in complete_markdown
    assert "--refresh-from" not in complete_markdown


def test_checked_in_workload_corpus_is_complete_and_correct() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "simulator_workload_corpus_cpu_arm64_20260924.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["workloads"] == list(WORKLOAD_NAMES[:-1])
    assert payload["n_wires"] == [10, 14, 18, 22]
    assert len(payload["cases"]) == 20
    assert payload["all_measurements_stable"] is False
    assert payload["methodology"]["measurement_scope"] == (
        "end_to_end_user_facing_run_call"
    )
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert set(case["engines"]) == {
            "flagquantum_native",
            "qiskit_aer",
            "cirq_simulator",
            "pennylane_lightning_qubit",
        }
        assert case["features"]["feature_fingerprint"]
        for engine in case["engines"].values():
            assert engine["end_to_end"]["sample_count"] == 9
            assert engine["end_to_end"]["median_seconds"] > 0

    qft_cases = {
        case["workload"]["n_wires"]: case
        for case in payload["cases"]
        if case["workload"]["name"] == "truncated_qft_statevector"
    }
    assert qft_cases[18]["engines"]["flagquantum_native"]["end_to_end"][
        "median_seconds"
    ] == pytest.approx(0.00804412504658103)
    assert qft_cases[22]["engines"]["flagquantum_native"]["end_to_end"][
        "median_seconds"
    ] == pytest.approx(0.1513523330213502)
    assert qft_cases[22]["engines"]["cirq_simulator"]["end_to_end"][
        "median_seconds"
    ] == pytest.approx(0.15457395801786333)
    assert all(
        ratio > 1
        for case in qft_cases.values()
        for ratio in case["comparison"]["engine_over_flagquantum_median"].values()
    )
    refreshed_cases = {
        (case["workload"]["name"], case["workload"]["n_wires"]): case
        for case in payload["cases"]
    }
    assert refreshed_cases[("random_clifford_statevector", 18)]["engines"][
        "flagquantum_native"
    ]["end_to_end"]["median_seconds"] == pytest.approx(0.02002929092850536)
    assert refreshed_cases[("dense_nonlocal_statevector", 22)]["engines"][
        "flagquantum_native"
    ]["end_to_end"]["median_seconds"] == pytest.approx(0.3745365421054885)
    assert refreshed_cases[("dense_nonlocal_statevector", 22)]["engines"]["qiskit_aer"][
        "end_to_end"
    ]["median_seconds"] == pytest.approx(1.9517207079916261)
    assert payload["refresh_history"][-1]["preserved_engines"] == [
        "qiskit_aer",
        "cirq_simulator",
        "pennylane_lightning_qubit",
    ]

    report_path = path.with_name("SIMULATOR_WORKLOAD_CORPUS_CPU_ARM64_20260924.md")
    assert report_path.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_swap_routing_case_is_complete_and_correct() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "simulator_swap_routing_cpu_arm64_20260924.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["workloads"] == ["swap_routing_statevector"]
    assert payload["n_wires"] == [22]
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert case["workload"]["gate_count"] == 150
    assert case["correctness"]["passed"] is True
    assert set(case["engines"]) == {
        "flagquantum_native",
        "qiskit_aer",
        "cirq_simulator",
        "pennylane_lightning_qubit",
    }
    for engine in case["engines"].values():
        assert engine["end_to_end"]["sample_count"] == 7
        assert engine["end_to_end"]["median_seconds"] > 0


def test_checked_in_dense_nonlocal_case_is_complete_stable_and_correct() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "simulator_dense_nonlocal_cpu_arm64_20260924.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["hostname"] == "redacted"
    assert payload["workloads"] == ["dense_nonlocal_statevector"]
    assert payload["n_wires"] == [22]
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert case["workload"]["gate_count"] == 275
    assert case["features"]["opcode_histogram"]["cz"] == 231
    assert case["correctness"]["passed"] is True
    assert set(case["engines"]) == {
        "flagquantum_native",
        "qiskit_aer",
        "cirq_simulator",
        "pennylane_lightning_qubit",
    }
    assert all(
        ratio > 1
        for ratio in case["comparison"]["engine_over_flagquantum_median"].values()
    )
    for engine in case["engines"].values():
        assert engine["end_to_end"]["sample_count"] == 7
        assert engine["end_to_end"]["median_seconds"] > 0


def test_checked_in_random_clifford_case_is_complete_stable_and_correct() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / "simulator_random_clifford_cpu_arm64_20260924.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["hostname"] == "redacted"
    assert payload["workloads"] == ["random_clifford_statevector"]
    assert payload["n_wires"] == [18, 22]
    assert len(payload["cases"]) == 2
    cases = {case["workload"]["n_wires"]: case for case in payload["cases"]}
    assert cases[18]["workload"]["gate_count"] == 108
    assert cases[22]["workload"]["gate_count"] == 132
    assert all(case["correctness"]["passed"] for case in cases.values())
    assert all(
        engine["end_to_end"]["sample_count"] == 9
        for case in cases.values()
        for engine in case["engines"].values()
    )
    assert all(
        ratio > 1
        for ratio in cases[22]["comparison"]["engine_over_flagquantum_median"].values()
    )


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"workloads": ()}, "at least one"),
        ({"n_wires": ()}, "at least one"),
        ({"threads": 0}, "threads must be positive"),
        ({"workloads": ("unknown",)}, "unsupported workload"),
    ),
)
def test_corpus_rejects_invalid_matrix(kwargs: dict[str, object], message: str) -> None:
    arguments = {
        "workloads": ("hardware_efficient_statevector",),
        "n_wires": (4,),
        "engines": ("flagquantum_native",),
        "threads": 1,
        "warmup": 0,
        "iterations": 3,
        "calls_per_sample": 1,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_benchmark(**arguments)
