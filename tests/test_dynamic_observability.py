from __future__ import annotations

import torch

import flagquantum as fq
from benchmarks.dynamic_trajectory import run_benchmark


def _feedback_circuit(*, bsz: int = 1) -> fq.experimental.DynamicCircuit:
    circuit = fq.experimental.DynamicCircuit(2, bsz=bsz)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0, equals=1)
    circuit.reset(0)
    return circuit


def test_dynamic_statistics_count_trajectories_operations_and_branches() -> None:
    result = fq.experimental.run_dynamic(_feedback_circuit(), shots=64, seed=19)
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
    assert result.provider_metadata["provider"] == "flagquantum"

    canonical = result.to_execution_result()
    assert canonical.runtime["dynamic_statistics"] == statistics


def test_dynamic_statistics_are_seed_reproducible_and_cover_batches() -> None:
    first = fq.experimental.run_dynamic(_feedback_circuit(bsz=2), shots=16, seed=5)
    second = fq.experimental.run_dynamic(_feedback_circuit(bsz=2), shots=16, seed=5)

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
