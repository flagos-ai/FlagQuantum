from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.batched_statevector_corpus import (
    SCHEMA,
    build_parameter_batch,
    render_markdown,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_runner_is_registered_and_lazy() -> None:
    assert "batched_statevector_corpus" in runners.names()
    assert callable(runners.resolve("batched_statevector_corpus"))


def test_independent_parameter_batch_matches_scalar_rows() -> None:
    batched, scalar = build_parameter_batch(
        "hardware_efficient_statevector", n_wires=4, batch_size=3
    )

    expected = torch.cat(tuple(circuit.state() for circuit in scalar), dim=0)
    torch.testing.assert_close(batched.state(), expected, atol=1e-10, rtol=1e-10)
    assert not torch.equal(expected[0], expected[1])


def test_small_native_corpus_records_task_throughput_and_correctness() -> None:
    payload = run_benchmark(
        workloads=("hardware_efficient_statevector",),
        n_wires=(4,),
        batch_sizes=(1, 3),
        engines=(
            "flagquantum_native_batch",
            "flagquantum_native_monolithic_batch",
            "flagquantum_native_serial",
        ),
        threads=1,
        warmup=0,
        iterations=3,
    )

    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["artifact_class"] == "measured_comparison_run"
    assert payload["scalability_claim_allowed"] is False
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert case["correctness"]["passed"] is True
        assert (
            case["correctness"]["engines"]["flagquantum_native_serial"]["max_abs_error"]
            <= 1e-10
        )
        assert (
            case["correctness"]["engines"]["flagquantum_native_monolithic_batch"][
                "max_abs_error"
            ]
            <= 1e-10
        )
        for result in case["engines"].values():
            assert result["batch_total"]["sample_count"] == 3
            assert result["batch_total"]["median_seconds"] > 0
            assert result["median_seconds_per_statevector"] > 0
            assert result["median_statevectors_per_second"] > 0

    report = render_markdown(payload, artifact_name="result.json")
    assert "FlagQuantum native batch" in report
    assert "repeatedly" in report
    assert "not each external framework's best raw batching API" in report


def test_invalid_batch_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_parameter_batch("hardware_efficient_statevector", n_wires=4, batch_size=0)
    with pytest.raises(ValueError, match="iterations at least 3"):
        run_benchmark(
            workloads=("hardware_efficient_statevector",),
            n_wires=(4,),
            batch_sizes=(1,),
            engines=("flagquantum_native_batch",),
            threads=1,
            warmup=0,
            iterations=2,
        )


def test_checked_in_measurement_artifacts_are_complete() -> None:
    comparison = REPOSITORY_ROOT / "benchmarks" / "results" / "comparison"
    main = json.loads(
        (comparison / "batched_statevector_corpus_cpu_arm64_20260930.json").read_text(
            encoding="utf-8"
        )
    )
    stability = json.loads(
        (
            comparison / "batched_statevector_corpus_stability_cpu_arm64_20260930.json"
        ).read_text(encoding="utf-8")
    )

    assert main["schema"] == SCHEMA
    assert main["correctness_passed"] is True
    assert main["scalability_claim_allowed"] is False
    assert len(main["cases"]) == 45
    assert stability["schema"] == SCHEMA
    assert stability["correctness_passed"] is True
    assert stability["all_measurements_stable"] is True
    assert len(stability["cases"]) == 4
