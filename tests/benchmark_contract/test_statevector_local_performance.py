from flagquantum.benchmarking.statevector_local import SCHEMA, run_benchmark


def test_local_statevector_performance_payload_is_correct_and_fail_closed():
    payload = run_benchmark(
        n_wires=4,
        batch_size=2,
        layers=1,
        device="cpu",
        warmup=0,
        iterations=2,
    )

    assert payload["schema_version"] == SCHEMA
    assert payload["artifact_class"] == "measured_local_run"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["correctness"]["passed"] is True
    assert payload["correctness"]["max_abs_error"] <= 2e-5
    assert payload["optimized"]["sample_count"] == 2
    assert payload["sequential_reference"]["sample_count"] == 2
    assert payload["optimized"]["runtime"]["fused_gate_count"] >= 3
    assert payload["optimized"]["runtime"]["permutation_gates"] >= 1
    assert payload["stability_gate_passed"] is (
        payload["optimized"]["coefficient_of_variation"]
        <= payload["regression_thresholds"]["max_coefficient_of_variation"]
    )
    assert payload["speedup"] > 0


def test_local_statevector_performance_rejects_non_repeated_measurement():
    try:
        run_benchmark(
            n_wires=4,
            batch_size=1,
            layers=1,
            device="cpu",
            warmup=0,
            iterations=1,
        )
    except ValueError as error:
        assert "iterations must be >= 2" in str(error)
    else:
        raise AssertionError("single-sample performance payload must fail closed")
