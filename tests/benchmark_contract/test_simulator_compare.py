from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from flagquantum.benchmarking.simulator_compare import (
    ABSOLUTE_TOLERANCE,
    RUNNER,
    SCHEMA,
    build_workload,
    run_benchmark,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[2]


def test_simulator_evaluation_contract_matches_runner() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts" / "simulator-evaluation-contract.toml").read_text()
    )

    assert contract["runner"] == RUNNER
    assert contract["result_schema"] == SCHEMA
    assert contract["correctness"]["absolute_tolerance"] == ABSOLUTE_TOLERANCE
    assert contract["engines"] == ["flagquantum_native", "qiskit_aer"]
    assert contract["hidden_fallback_allowed"] is False
    assert contract["scalability_claim_allowed"] is False
    assert contract["matrix"] == {
        "standard_n_wires": [10, 14, 18, 22, 24],
        "small_latency_n_wires": [10, 14],
        "transition_n_wires": [18, 22],
        "memory_pressure_n_wires": [24],
        "capacity_probe_minimum_n_wires": 26,
        "capacity_probe_release_gating": False,
        "small_width_only_claims_whole_simulator_performance": False,
    }


def test_simulator_workload_is_deterministic_and_dense() -> None:
    first = build_workload(n_wires=5, layers=2)
    second = build_workload(n_wires=5, layers=2)

    assert first.to_ir().content_hash == second.to_ir().content_hash
    assert len(first) == 42
    assert first.dtype == torch.complex128


def test_simulator_comparison_rejects_too_few_iterations() -> None:
    with pytest.raises(ValueError, match="iterations at least 3"):
        run_benchmark(
            n_wires=(4,),
            layers=1,
            threads=1,
            warmup=0,
            iterations=2,
            setup_iterations=1,
            calls_per_sample=1,
        )


def test_simulator_comparison_rejects_an_empty_size_matrix() -> None:
    with pytest.raises(ValueError, match="at least one workload size"):
        run_benchmark(
            n_wires=(),
            layers=1,
            threads=1,
            warmup=0,
            iterations=3,
            setup_iterations=1,
            calls_per_sample=1,
        )


@pytest.mark.qiskit
def test_simulator_comparison_smoke_payload() -> None:
    pytest.importorskip("qiskit_aer")

    payload = run_benchmark(
        n_wires=(4,),
        layers=1,
        threads=1,
        warmup=0,
        iterations=3,
        setup_iterations=1,
        calls_per_sample=1,
    )

    assert payload["passed"] is True
    assert payload["correctness_passed"] is True
    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["claim_evidence_type"] == "unknown"
    assert payload["non_release_evidence"] is True
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_blockers"] == (
        "comparison_result_not_release_scalability_evidence",
    )
    assert payload["methodology"]["steady_state_order"] == (
        "alternating_within_one_process"
    )
    case = payload["cases"][0]
    assert case["correctness"]["max_abs_error"] <= ABSOLUTE_TOLERANCE
    assert set(case["engines"]) == {"flagquantum_native", "qiskit_aer"}
    assert case["comparison"]["steady_state_qiskit_over_flagquantum"] > 0
