from __future__ import annotations

import pytest

import flagquantum.benchmarking as runners
import flagquantum.benchmarking.socket_local_throughput as socket_throughput
from flagquantum.benchmarking.socket_local_throughput import (
    latency_summary,
    parse_cpu_list,
    render_markdown,
    split_cpu_groups,
)

pytestmark = pytest.mark.benchmark_contract


def test_socket_local_runner_is_registered_lazily() -> None:
    assert "socket_local_throughput" in runners.names()
    spec = runners.describe("socket_local_throughput")
    assert "physical socket CPU list" in spec.hardware
    assert callable(runners.resolve("socket_local_throughput"))


def test_cpu_list_and_worker_groups_are_exact_and_disjoint() -> None:
    cpus = parse_cpu_list("0-3,8,10-12")
    assert cpus == (0, 1, 2, 3, 8, 10, 11, 12)
    groups = split_cpu_groups(cpus, 4)
    assert groups == ((0, 1), (2, 3), (8, 10), (11, 12))
    assert set().union(*map(set, groups)) == set(cpus)


@pytest.mark.parametrize("value", ("", "0,,1", "2-1", "0,0", "-1"))
def test_invalid_cpu_lists_fail_closed(value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_cpu_list(value)


def test_latency_summary_reports_interpolated_p95_and_stability() -> None:
    payload = latency_summary((1.0, 2.0, 3.0, 4.0, 5.0))

    assert payload["sample_count"] == 5
    assert payload["p50_seconds"] == 3.0
    assert payload["p95_seconds"] == pytest.approx(4.8)
    assert payload["relative_median_absolute_deviation"] == pytest.approx(1 / 3)


def test_markdown_uses_absolute_throughput_tail_latency_and_speedup() -> None:
    def result(throughput: float, p50: float, p95: float) -> dict[str, object]:
        return {
            "tasks_per_second": throughput,
            "latency": {"p50_seconds": p50, "p95_seconds": p95},
        }

    payload = {
        "engines": ("flagquantum_native", "qiskit_aer"),
        "cases": (
            {
                "workload": "random_clifford_statevector",
                "configuration": {"workers": 1, "threads_per_worker": 4},
                "engines": {
                    "flagquantum_native": result(10.0, 0.1, 0.2),
                    "qiskit_aer": result(20.0, 0.05, 0.08),
                },
            },
            {
                "workload": "random_clifford_statevector",
                "configuration": {"workers": 2, "threads_per_worker": 2},
                "engines": {
                    "flagquantum_native": result(30.0, 0.06, 0.09),
                    "qiskit_aer": result(40.0, 0.04, 0.07),
                },
            },
        ),
    }

    document = render_markdown(payload, artifact_name="result.json")

    assert "30.000" in document
    assert "60.000 ms" in document
    assert "90.000 ms" in document
    assert "3.000x" in document
    assert "not single-circuit" in document


def test_worker_matrix_rejects_oversubscription_and_unequal_groups() -> None:
    with pytest.raises(ValueError, match="divide"):
        split_cpu_groups((0, 1, 2, 3), 3)
    with pytest.raises(ValueError, match="positive"):
        split_cpu_groups((0, 1), 0)


def test_independent_socket_throughput_is_not_scalability_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "tasks_per_second": 10.0,
        "stable": True,
        "correctness": {"passed": True},
    }
    monkeypatch.setattr(socket_throughput, "_validate_matrix", lambda *args: None)
    monkeypatch.setattr(socket_throughput, "run_configuration", lambda **kwargs: result)
    monkeypatch.setattr(
        socket_throughput, "_engine_versions", lambda engine: {engine: "test"}
    )

    payload = socket_throughput.run_benchmark(
        workloads=("random_clifford_statevector",),
        n_qubits=16,
        engines=("flagquantum_native",),
        cpus=(0,),
        worker_counts=(1,),
        total_tasks=1,
        warmup=0,
    )

    assert payload["distribution_semantics"] == (
        "single_socket_replicated_process_throughput"
    )
    assert payload["scalability_claim_allowed"] is False
    assert payload["scalability_blockers"]
