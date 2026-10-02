from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flagquantum.benchmarking.differentiable_simulator_corpus import (
    ADJOINT_ENGINE_NAMES,
    ENGINE_NAMES,
    SCHEMA,
    WORKLOAD_NAMES,
    _render_markdown,
    _timing_semantics,
    build_workload,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).parents[2]
RESULT_NAME = "differentiable_simulator_corpus_cpu_arm64_20260928.json"
ADJOINT_RESULT_NAME = "adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json"
FORWARD_CX_RESULT_NAME = "native_cpu_forward_cx_gather_cpu_arm64_20260929.json"
OBSERVABLE_CACHE_RESULT_NAME = (
    "native_cpu_adjoint_observable_cache_cpu_arm64_20260929.json"
)
ROTATION_TILE_RESULT_NAME = "native_cpu_adjoint_rotation_tiles_cpu_arm64_20260929.json"
EULER_TRIPLE_RESULT_NAME = "native_cpu_adjoint_euler_triples_cpu_arm64_20260929.json"
OBSERVABLE_BOUNDARY_RESULT_NAME = (
    "native_cpu_adjoint_observable_boundary_cpu_arm64_20260929.json"
)
SHARED_RZZ_RESULT_NAME = "native_cpu_shared_rzz_cpu_arm64_20260929.json"
FORWARD_RZZ_ROTATION_RESULT_NAME = (
    "native_cpu_forward_rzz_rotation_cpu_arm64_20260929.json"
)
OBSERVABLE_ROTATION_RESULT_NAME = (
    "native_cpu_observable_rotation_cpu_arm64_20260929.json"
)
FORWARD_WIDE_TILE_RESULT_NAME = (
    "native_cpu_forward_wide_rotation_tiles_cpu_arm64_20260929.json"
)
TERMINAL_NO_RESTORE_RESULT_NAME = (
    "native_cpu_adjoint_terminal_no_restore_cpu_arm64_20260929.json"
)
COMPACT_CX_RESULT_NAME = "native_cpu_adjoint_compact_cx_cpu_arm64_20260930.json"
SAVED_PARAMETER_VALIDATION_RESULT_NAME = (
    "native_cpu_adjoint_saved_parameter_validation_cpu_arm64_20261002.json"
)
PARALLEL_GRAIN_RESULT_NAME = "native_cpu_adjoint_parallel_grain_cpu_arm64_20261002.json"


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
    assert payload["methodology"]["primary_performance_metric"] == "value_and_grad"
    assert payload["methodology"]["cross_engine_phase_comparison_allowed"] is False
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        engine = case["engines"]["flagquantum_native"]
        assert engine["timing_semantics"]["cross_engine_comparable_metrics"] == [
            "value_and_grad"
        ]
        assert (
            engine["timing_semantics"]["cross_engine_phase_comparison_allowed"] is False
        )
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


def test_lightning_timing_semantics_prevent_phase_ranking() -> None:
    semantics = _timing_semantics("pennylane_lightning_adjoint")

    assert semantics["primary_performance_metric"] == "value_and_grad"
    assert semantics["cross_engine_comparable_metrics"] == ["value_and_grad"]
    assert semantics["cross_engine_phase_comparison_allowed"] is False
    assert semantics["gradient_work_attribution"] == (
        "adjoint_derivative_work_may_run_during_value_evaluation"
    )


def test_forward_cx_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_CX_GATHER", "custom")
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe",),
        n_wires=(4,),
        layers=1,
        engines=(
            "flagquantum_adjoint",
            "flagquantum_adjoint_forward_cx_rollback",
        ),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ["FQ_NATIVE_CPU_CX_GATHER"] == "custom"
    support = payload["support_matrix"]["flagquantum_adjoint_forward_cx_rollback"]
    assert support["included"] is True
    assert "forward CX gather disabled" in support["contract"]


def test_observable_cache_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE", "custom")
    payload = run_benchmark(
        workloads=("qaoa_path_maxcut",),
        n_wires=(4,),
        layers=1,
        engines=(
            "flagquantum_adjoint",
            "flagquantum_adjoint_observable_cache_rollback",
        ),
        threads=1,
        warmup=1,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ["FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE"] == "custom"
    support = payload["support_matrix"]["flagquantum_adjoint_observable_cache_rollback"]
    assert support["included"] is True
    assert "observable-weight cache disabled" in support["contract"]


def test_rotation_tile_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "custom")
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe",),
        n_wires=(4,),
        layers=1,
        engines=(
            "flagquantum_adjoint",
            "flagquantum_adjoint_rotation_tile_rollback",
        ),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ["FQ_NATIVE_CPU_ADJOINT_WIDE_TILES"] == "custom"
    support = payload["support_matrix"]["flagquantum_adjoint_rotation_tile_rollback"]
    assert support["included"] is True
    assert "legacy two-wire rotation tiles" in support["contract"]


def test_euler_triple_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES", "custom")
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe",),
        n_wires=(4,),
        layers=1,
        engines=(
            "flagquantum_adjoint",
            "flagquantum_adjoint_euler_triple_rollback",
        ),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ["FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES"] == "custom"
    support = payload["support_matrix"]["flagquantum_adjoint_euler_triple_rollback"]
    assert support["included"] is True
    assert "Euler fast path disabled" in support["contract"]


def test_observable_boundary_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY", "custom")
    rollback = "flagquantum_adjoint_observable_boundary_rollback"
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ["FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY"] == "custom"
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert "eager observable boundary" in support["contract"]


def test_shared_rzz_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "FQ_NATIVE_CPU_ADJOINT_RX_RZZ_FUSION"
    monkeypatch.setenv(variable, "custom")
    rollback = "flagquantum_adjoint_shared_rzz_rollback"
    payload = run_benchmark(
        workloads=("qaoa_path_maxcut",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ[variable] == "custom"
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert "shared-RZZ fusion disabled" in support["contract"]


def test_observable_rotation_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "FQ_NATIVE_CPU_OBSERVABLE_ROTATION_BOUNDARY"
    monkeypatch.setenv(variable, "custom")
    rollback = "flagquantum_adjoint_observable_rotation_rollback"
    payload = run_benchmark(
        workloads=("qaoa_path_maxcut",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ[variable] == "custom"
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert "observable seeding separate" in support["contract"]


def test_forward_rzz_rotation_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = (
        "FQ_NATIVE_CPU_ADJOINT_RZZ_H_FUSION",
        "FQ_NATIVE_CPU_SHARED_RZZ_FORWARD_FUSION",
        "FQ_NATIVE_CPU_FORWARD_SPECIALIZED_ROTATIONS",
    )
    for variable in variables:
        monkeypatch.setenv(variable, "custom")
    rollback = "flagquantum_adjoint_forward_rzz_rotation_rollback"
    payload = run_benchmark(
        workloads=("qaoa_path_maxcut",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert all(os.environ[variable] == "custom" for variable in variables)
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert "backward RZZ/H boundary fusion" in support["contract"]


def test_forward_wide_tile_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "FQ_NATIVE_CPU_FORWARD_WIDE_ROTATION_TILES"
    monkeypatch.setenv(variable, "custom")
    rollback = "flagquantum_adjoint_forward_wide_tile_rollback"
    payload = run_benchmark(
        workloads=("qaoa_path_maxcut",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ[variable] == "custom"
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert "legacy six-wire forward rotation tiles" in support["contract"]


def test_flat_pair_simd_rollback_engine_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "FQ_NATIVE_CPU_ADJOINT_FLAT_PAIR_SIMD"
    monkeypatch.setenv(variable, "custom")
    rollback = "flagquantum_adjoint_flat_pair_simd_rollback"
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ[variable] == "custom"
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert "nested rotation-pair traversal" in support["contract"]


@pytest.mark.parametrize(
    ("variable", "rollback", "contract"),
    (
        (
            "FQ_NATIVE_CPU_ADJOINT_CX_ROTATION_FUSION",
            "flagquantum_adjoint_cx_rotation_fusion_rollback",
            "separate CX gather",
        ),
        (
            "FQ_NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE",
            "flagquantum_adjoint_terminal_no_restore_rollback",
            "state restoration enabled",
        ),
        (
            "FQ_NATIVE_CPU_COMPACT_CX_INDEX",
            "flagquantum_adjoint_compact_cx_index_rollback",
            "materialized full-state CX index",
        ),
    ),
)
def test_adjoint_layer_rollback_engines_restore_environment(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    rollback: str,
    contract: str,
) -> None:
    monkeypatch.setenv(variable, "custom")
    payload = run_benchmark(
        workloads=("hardware_efficient_vqe",),
        n_wires=(4,),
        layers=1,
        engines=("flagquantum_adjoint", rollback),
        threads=1,
        warmup=0,
        iterations=3,
        calls_per_sample=1,
        measure_memory=False,
    )

    assert payload["passed"] is True
    assert os.environ[variable] == "custom"
    support = payload["support_matrix"][rollback]
    assert support["included"] is True
    assert contract in support["contract"]


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


def test_checked_in_forward_cx_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / FORWARD_CX_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["non_release_evidence"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_blockers"]
    assert payload["engines"] == [
        "flagquantum_adjoint",
        "flagquantum_adjoint_forward_cx_rollback",
    ]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 8
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert case["engines"]["flagquantum_adjoint"]["forward"]["sample_count"] == 11

    report = path.with_name("NATIVE_CPU_FORWARD_CX_GATHER_CPU_ARM64_20260929.md")
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_observable_cache_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / OBSERVABLE_CACHE_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["non_release_evidence"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_blockers"]
    assert payload["engines"] == [
        "flagquantum_adjoint",
        "flagquantum_adjoint_observable_cache_rollback",
    ]
    assert payload["environment"]["torch_threads"] == 8
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert (
            case["comparison"]["engine_over_flagquantum_median"][
                "flagquantum_adjoint_observable_cache_rollback"
            ]["forward"]
            > 1.0
        )

    report = path.with_name("NATIVE_CPU_ADJOINT_OBSERVABLE_CACHE_CPU_ARM64_20260929.md")
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_rotation_tile_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / ROTATION_TILE_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["non_release_evidence"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["engines"] == [
        "flagquantum_adjoint",
        "flagquantum_adjoint_rotation_tile_rollback",
    ]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 8
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 11

    vqe = payload["cases"][0]
    ratio = vqe["comparison"]["engine_over_flagquantum_median"][
        "flagquantum_adjoint_rotation_tile_rollback"
    ]
    assert ratio["backward"] > 1.3
    assert ratio["value_and_grad"] > 1.2

    report = path.with_name("NATIVE_CPU_ADJOINT_ROTATION_TILES_CPU_ARM64_20260929.md")
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_euler_triple_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / EULER_TRIPLE_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["non_release_evidence"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["engines"] == [
        "flagquantum_adjoint",
        "flagquantum_adjoint_euler_triple_rollback",
    ]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 8
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 11

    vqe = payload["cases"][0]
    ratio = vqe["comparison"]["engine_over_flagquantum_median"][
        "flagquantum_adjoint_euler_triple_rollback"
    ]
    assert ratio["backward"] > 1.15
    assert ratio["value_and_grad"] > 1.10

    report = path.with_name("NATIVE_CPU_ADJOINT_EULER_TRIPLES_CPU_ARM64_20260929.md")
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_observable_boundary_comparison_is_reproducible() -> None:
    path = (
        ROOT / "benchmarks" / "results" / "comparison" / OBSERVABLE_BOUNDARY_RESULT_NAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_observable_boundary_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == ["flagquantum_adjoint", rollback]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 4
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 9

    vqe = payload["cases"][0]
    ratio = vqe["comparison"]["engine_over_flagquantum_median"][rollback]
    assert ratio["forward"] > 1.10
    assert ratio["backward"] > 1.05
    assert ratio["value_and_grad"] > 1.05

    report = path.with_name(
        "NATIVE_CPU_ADJOINT_OBSERVABLE_BOUNDARY_CPU_ARM64_20260929.md"
    )
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_shared_rzz_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / SHARED_RZZ_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_shared_rzz_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == ["flagquantum_adjoint", rollback]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 4
    case = payload["cases"][0]
    assert case["workload"]["name"] == "qaoa_path_maxcut"
    assert case["correctness"]["passed"] is True
    assert case["stability"]["passed"] is True
    assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 9
    ratio = case["comparison"]["engine_over_flagquantum_median"][rollback]
    assert ratio["forward"] > 1.09
    assert ratio["backward"] > 1.03
    assert ratio["value_and_grad"] > 1.08

    report = path.with_name("NATIVE_CPU_SHARED_RZZ_CPU_ARM64_20260929.md")
    text = report.read_text(encoding="utf-8")
    assert "180.598" in text
    assert "196.709" in text
    assert "1.09x" in text


def test_checked_in_forward_rzz_rotation_comparison_is_reproducible() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / FORWARD_RZZ_ROTATION_RESULT_NAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_forward_rzz_rotation_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == ["flagquantum_adjoint", rollback]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 4
    case = payload["cases"][0]
    assert case["workload"]["name"] == "qaoa_path_maxcut"
    assert case["correctness"]["passed"] is True
    assert case["stability"]["passed"] is True
    assert case["engines"]["flagquantum_adjoint"]["forward"]["sample_count"] == 31
    ratio = case["comparison"]["engine_over_flagquantum_median"][rollback]
    assert ratio["forward"] > 1.80
    assert ratio["backward"] > 1.40
    assert ratio["value_and_grad"] > 1.55

    report = path.with_name("NATIVE_CPU_FORWARD_RZZ_ROTATION_CPU_ARM64_20260929.md")
    text = report.read_text(encoding="utf-8")
    assert "66.208" in text
    assert "65.829" in text
    assert "1.45x" in text
    assert "1.59x" in text


def test_checked_in_observable_rotation_comparison_is_reproducible() -> None:
    path = (
        ROOT / "benchmarks" / "results" / "comparison" / OBSERVABLE_ROTATION_RESULT_NAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_observable_rotation_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == [
        "flagquantum_adjoint",
        rollback,
        "pennylane_lightning_adjoint",
    ]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 4
    case = payload["cases"][0]
    assert case["correctness"]["passed"] is True
    assert case["stability"]["passed"] is True
    assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 31
    ratios = case["comparison"]["engine_over_flagquantum_median"]
    assert ratios[rollback]["backward"] > 1.02
    assert ratios["pennylane_lightning_adjoint"]["value_and_grad"] > 6.0

    report = path.with_name("NATIVE_CPU_OBSERVABLE_ROTATION_CPU_ARM64_20260929.md")
    text = report.read_text(encoding="utf-8")
    assert "Value evaluation (ms)" in text
    assert "Autograd callback (ms)" in text
    assert "Only value + gradient is" in text
    assert "comparable across engines" in text
    assert "Backward ratio vs optimized" not in text
    assert "69.109" in text
    assert "70.938" in text
    assert "913.521" in text
    assert "6.45x" in text


def test_checked_in_forward_wide_tile_comparison_is_reproducible() -> None:
    path = (
        ROOT / "benchmarks" / "results" / "comparison" / FORWARD_WIDE_TILE_RESULT_NAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_forward_wide_tile_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == ["flagquantum_adjoint", rollback]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 2
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert case["engines"]["flagquantum_adjoint"]["forward"]["sample_count"] == 41
        ratio = case["comparison"]["engine_over_flagquantum_median"][rollback]
        assert ratio["forward"] > 1.02
        assert ratio["value_and_grad"] > 1.0

    report = path.with_name(
        "NATIVE_CPU_FORWARD_WIDE_ROTATION_TILES_CPU_ARM64_20260929.md"
    )
    assert report.read_text(encoding="utf-8") == _render_markdown(
        payload, artifact_name=path.name
    )


def test_checked_in_terminal_no_restore_comparison_is_reproducible() -> None:
    path = (
        ROOT / "benchmarks" / "results" / "comparison" / TERMINAL_NO_RESTORE_RESULT_NAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_terminal_no_restore_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == ["flagquantum_adjoint", rollback]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 2
    case = payload["cases"][0]
    assert case["correctness"]["passed"] is True
    assert case["stability"]["passed"] is True
    assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 41
    ratio = case["comparison"]["engine_over_flagquantum_median"][rollback]
    assert ratio["backward"] > 1.35
    assert ratio["value_and_grad"] > 1.20

    report = path.with_name(
        "NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE_CPU_ARM64_20260929.md"
    ).read_text(encoding="utf-8")
    assert "98.069" in report
    assert "137.573" in report
    assert "1.403x" in report
    assert "FlagQuantum example" in report
    assert "## Reproduce" in report


def test_checked_in_compact_cx_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / COMPACT_CX_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_compact_cx_index_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == [
        "flagquantum_adjoint",
        rollback,
        "pennylane_lightning_adjoint",
    ]
    assert payload["n_wires"] == [22]
    assert payload["environment"]["torch_threads"] == 2
    case = payload["cases"][0]
    assert case["correctness"]["passed"] is True
    assert case["stability"]["passed"] is True
    assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 41
    assert case["engines"]["flagquantum_adjoint"]["memory"]["peak_rss_bytes"] < (
        case["engines"][rollback]["memory"]["peak_rss_bytes"]
    )

    report = path.with_name(
        "NATIVE_CPU_ADJOINT_COMPACT_CX_CPU_ARM64_20260930.md"
    ).read_text(encoding="utf-8")
    assert "158.922" in report
    assert "155.526" in report
    assert "600.766" in report
    assert "699.312" in report
    assert "95,325x" in report
    assert "PennyLane Lightning adjoint" in report
    assert "FlagQuantum example" in report
    assert "## Reproduce" in report


def test_checked_in_saved_parameter_validation_comparison_is_reproducible() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "results"
        / "comparison"
        / SAVED_PARAMETER_VALIDATION_RESULT_NAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_saved_parameter_revalidation"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == [
        "flagquantum_adjoint",
        rollback,
        "pennylane_lightning_adjoint",
    ]
    assert payload["n_wires"] == [10, 14, 18, 22]
    assert payload["environment"]["torch_threads"] == 1
    assert len(payload["cases"]) == 8
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        assert case["engines"]["flagquantum_adjoint"]["backward"]["sample_count"] == 21
        optimized = case["engines"]["flagquantum_adjoint"]
        revalidated = case["engines"][rollback]
        assert (
            optimized["backward"]["median_seconds"]
            <= 1.05 * revalidated["backward"]["median_seconds"]
        )
        assert (
            optimized["value_and_grad"]["median_seconds"]
            <= 1.05 * revalidated["value_and_grad"]["median_seconds"]
        )
        lightning = case["engines"]["pennylane_lightning_adjoint"]
        assert (
            lightning["value_and_grad"]["median_seconds"]
            > optimized["value_and_grad"]["median_seconds"]
        )

    hardware_small = [
        case
        for case in payload["cases"]
        if case["workload"]["name"] == "hardware_efficient_vqe"
        and case["workload"]["n_wires"] in {10, 14}
    ]
    assert all(
        case["comparison"]["engine_over_flagquantum_median"][rollback]["backward"]
        > 1.10
        for case in hardware_small
    )

    report = path.with_name(
        "NATIVE_CPU_ADJOINT_SAVED_PARAMETER_VALIDATION_CPU_ARM64_20261002.md"
    ).read_text(encoding="utf-8")
    assert "1.209x" in report
    assert "4.774x" in report
    assert "not compared with FlagQuantum backward" in report
    assert "FlagQuantum example" in report
    assert "## Reproduce" in report


def test_checked_in_parallel_grain_comparison_is_reproducible() -> None:
    path = ROOT / "benchmarks" / "results" / "comparison" / PARALLEL_GRAIN_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollback = "flagquantum_adjoint_parallel_grain_rollback"

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["all_measurements_stable"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["scalability_claim_allowed"] is False
    assert payload["engines"] == [
        "flagquantum_adjoint",
        rollback,
        "pennylane_lightning_adjoint",
    ]
    assert payload["n_wires"] == [14, 16, 18, 20, 22]
    assert payload["environment"]["torch_threads"] == 8
    assert len(payload["cases"]) == 10
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert case["stability"]["passed"] is True
        optimized = case["engines"]["flagquantum_adjoint"]
        legacy = case["engines"][rollback]
        lightning = case["engines"]["pennylane_lightning_adjoint"]
        assert optimized["backward"]["sample_count"] == 21
        assert (
            optimized["value_and_grad"]["median_seconds"]
            <= 1.05 * legacy["value_and_grad"]["median_seconds"]
        )
        assert (
            lightning["value_and_grad"]["median_seconds"]
            > optimized["value_and_grad"]["median_seconds"]
        )

    medium_cases = [
        case for case in payload["cases"] if case["workload"]["n_wires"] in {16, 18}
    ]
    assert all(
        case["comparison"]["engine_over_flagquantum_median"][rollback]["backward"]
        > 1.25
        for case in medium_cases
    )
    assert all(
        case["comparison"]["engine_over_flagquantum_median"][rollback]["value_and_grad"]
        > 1.10
        for case in medium_cases
    )

    report = path.with_name(
        "NATIVE_CPU_ADJOINT_PARALLEL_GRAIN_CPU_ARM64_20261002.md"
    ).read_text(encoding="utf-8")
    assert "2.272x" in report
    assert "18.007x" in report
    assert "PennyLane Lightning" in report
    assert "FlagQuantum example" in report
    assert "## Reproduce" in report


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
