from __future__ import annotations

import json
from pathlib import Path

import pytest

from flagquantum.benchmarking.differentiable_simulator_corpus import (
    ADJOINT_ENGINE_NAMES,
    ENGINE_NAMES,
    SCHEMA,
    WORKLOAD_NAMES,
    _render_markdown,
    build_workload,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]
RESULT_NAME = "differentiable_simulator_corpus_cpu_arm64_20260928.json"
ADJOINT_RESULT_NAME = "adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json"


def test_differentiable_workloads_have_declared_structure() -> None:
    vqe = build_workload("hardware_efficient_vqe", n_wires=4, layers=1)
    qaoa = build_workload("qaoa_path_maxcut", n_wires=4, layers=1)

    assert len(vqe.circuit.to_ir().instructions) == 15
    assert vqe.parameters.shape == (1, 4, 3)
    assert vqe.hamiltonian.n_terms == 7
    assert len(qaoa.circuit.to_ir().instructions) == 11
    assert qaoa.parameters.shape == (1, 2)
    assert qaoa.hamiltonian.n_terms == 3
    assert qaoa.constant == 1.5


def test_native_differentiable_corpus_records_forward_and_backward() -> None:
    payload = run_benchmark(
        workloads=WORKLOAD_NAMES,
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_native",),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["methodology"]["optimizer_step_included"] is False
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        engine = case["engines"]["flagquantum_native"]
        assert engine["forward"]["sample_count"] == 3
        assert engine["backward"]["median_seconds"] > 0
        assert engine["value_and_grad"]["median_seconds"] > 0
        assert engine["memory"] == {
            "peak_rss_bytes": None,
            "scope": "not_measured",
        }


def test_native_adjoint_corpus_records_method_matched_gradient() -> None:
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe", "qaoa_path_maxcut"),
        n_wires=(4,),
        layers=1,
        engines=(ADJOINT_ENGINE_NAMES[0],),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    for case in payload["cases"]:
        engine = case["engines"]["flagquantum_adjoint"]
        assert engine["differentiation"] == "flagquantum_statevector_adjoint"
        assert engine["value_and_grad"]["sample_count"] == 3


def test_checked_in_differentiable_corpus_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["workloads"] == list(WORKLOAD_NAMES)
    assert payload["n_wires"] == [10, 14, 18, 22]
    assert payload["layers"] == 1
    assert payload["engines"] == list(ENGINE_NAMES)
    assert len(payload["cases"]) == 8
    assert payload["support_matrix"]["pennylane_default_qubit"]["gradient_method"] == (
        "torch_reverse_mode_autograd"
    )
    assert payload["support_matrix"]["qiskit_aer"]["included"] is False
    assert payload["support_matrix"]["cirq_simulator"]["included"] is False
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert set(case["engines"]) == set(ENGINE_NAMES)
        for engine in case["engines"].values():
            assert engine["forward"]["sample_count"] == 9
            assert engine["backward"]["median_seconds"] > 0
            assert engine["value_and_grad"]["median_seconds"] > 0
            assert engine["memory"]["peak_rss_bytes"] > 0

    report = path.with_name("DIFFERENTIABLE_SIMULATOR_CORPUS_CPU_ARM64_20260928.md")
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_adjoint_corpus_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / ADJOINT_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["engines"] == list(ADJOINT_ENGINE_NAMES)
    assert payload["n_wires"] == [10, 14, 18, 22]
    assert len(payload["cases"]) == 8
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert set(case["engines"]) == set(ADJOINT_ENGINE_NAMES)
        for engine in case["engines"].values():
            assert engine["value_and_grad"]["sample_count"] == 7
            assert engine["memory"]["peak_rss_bytes"] is None

    report = path.with_name(
        "ADJOINT_DIFFERENTIABLE_SIMULATOR_CORPUS_CPU_ARM64_20260928.md"
    )
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"workloads": ()}, "at least one"),
        ({"n_wires": ()}, "at least one"),
        ({"threads": 0}, "threads must be positive"),
        ({"layers": 0}, "layers must be positive"),
        ({"workloads": ("unknown",)}, "unsupported workload"),
    ),
)
def test_differentiable_corpus_rejects_invalid_matrix(
    kwargs: dict[str, object], message: str
) -> None:
    arguments = {
        "workloads": ("hardware_efficient_vqe",),
        "n_wires": (4,),
        "layers": 1,
        "engines": ("flagquantum_native",),
        "threads": 1,
        "warmup": 0,
        "iterations": 3,
        "calls_per_sample": 1,
        "measure_memory": False,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_benchmark(**arguments)
