from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from benchmarks.dynamic_trajectory import run_benchmark


def _feedback_circuit(*, bsz: int = 1) -> fq.experimental.dynamic.DynamicCircuit:
    circuit = fq.experimental.dynamic.DynamicCircuit(2, bsz=bsz)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0, equals=1)
    circuit.reset(0)
    return circuit


def test_dynamic_statistics_count_trajectories_operations_and_branches() -> None:
    result = fq.experimental.dynamic.run_dynamic(_feedback_circuit(), shots=64, seed=19)
    statistics = result.statistics

    assert statistics["trajectory_count"] == 64
    assert statistics["measurement_count"] == 64
    assert statistics["reset_count"] == 64
    assert (
        statistics["conditional_applied_count"]
        + statistics["conditional_skipped_count"]
        == 64
    )
    assert statistics["branch_count"] == len(statistics["branch_shots"])
    assert sum(statistics["branch_shots"].values()) == 64
    assert statistics["elapsed_seconds"] >= 0
    assert statistics["gate_execution_strategy"] == "batched_statevector_kernel"
    assert result.provider_metadata["provider"] == "flagquantum"

    canonical = result.to_execution_result()
    assert canonical.runtime["dynamic_statistics"] == statistics


def test_dynamic_statistics_are_seed_reproducible_and_cover_batches() -> None:
    first = fq.experimental.dynamic.run_dynamic(
        _feedback_circuit(bsz=2), shots=16, seed=5
    )
    second = fq.experimental.dynamic.run_dynamic(
        _feedback_circuit(bsz=2), shots=16, seed=5
    )

    assert torch.equal(first.samples, second.samples)
    assert torch.equal(first.classical_bits, second.classical_bits)
    assert first.statistics["trajectory_count"] == 32
    assert first.statistics["measurement_count"] == 32
    assert first.statistics["branch_shots"] == second.statistics["branch_shots"]
    assert sum(first.statistics["branch_shots"].values()) == 32


def test_dynamic_development_benchmark_contract_and_smoke_budget() -> None:
    payload = run_benchmark(
        shots=(32,),
        mid_circuit_measurements=(1,),
        backends=("flagquantum",),
        seed=3,
    )
    row = payload["rows"][0]

    assert payload["schema"] == "flagquantum_dynamic_trajectory_benchmark_v1"
    assert payload["artifact_classification"] == "development_microbenchmark"
    assert payload["scalability_claim_allowed"] is False
    assert row["shots"] == 32
    assert row["mid_circuit_measurements"] == 1
    assert row["shots_per_second"] > 0
    assert row["runtime_statistics"]["trajectory_count"] == 32
    assert row["elapsed_seconds"] < 10


def test_direct_dynamic_gate_path_matches_static_statevector_gates() -> None:
    dynamic = fq.experimental.dynamic.DynamicCircuit(3)
    dynamic.x(0).rx(1, theta=0.31).cx(0, 2).rzz(1, 2, theta=-0.27)
    dynamic.measure(0, classical_bit=0)
    result = fq.experimental.dynamic.run_dynamic(dynamic, shots=8, seed=13)

    static = fq.Circuit(3)
    static.x(0).rx(1, theta=0.31).cx(0, 2).rzz(1, 2, theta=-0.27)
    expected = static.state()[0]

    assert torch.allclose(result.final_states, expected.expand(8, -1))


def test_batched_and_reference_dynamic_strategies_are_semantically_equivalent() -> None:
    circuit = fq.experimental.dynamic.DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    reference = fq.experimental.dynamic.run_dynamic(
        circuit, shots=2048, seed=23, strategy="trajectory"
    )
    batched = fq.experimental.dynamic.run_dynamic(
        circuit, shots=2048, seed=23, strategy="batched"
    )

    for result in (reference, batched):
        assert torch.equal(result.samples[:, 0], result.classical_bits[:, 0])
        assert torch.equal(result.samples[:, 1], result.classical_bits[:, 0])
        probability = float(result.classical_bits[:, 0].float().mean())
        assert abs(probability - 0.5) < 0.05
    assert (
        reference.statistics["gate_execution_strategy"]
        == "trajectory_direct_statevector_kernel"
    )
    assert batched.statistics["gate_execution_strategy"] == "batched_statevector_kernel"


def test_auto_dynamic_strategy_falls_back_when_memory_budget_is_too_small() -> None:
    circuit = _feedback_circuit()
    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=64,
        seed=3,
        max_batched_bytes=1,
    )

    assert (
        result.statistics["gate_execution_strategy"]
        == "trajectory_direct_statevector_kernel"
    )
    with pytest.raises(ValueError, match="memory budget exceeded"):
        fq.experimental.dynamic.run_dynamic(
            circuit,
            shots=64,
            seed=3,
            strategy="batched",
            max_batched_bytes=1,
        )
