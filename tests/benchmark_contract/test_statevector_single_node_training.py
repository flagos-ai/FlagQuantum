from benchmarks.statevector_single_node_training import _classify_capacity_outcome


def test_expected_oom_is_explicit_capacity_failure() -> None:
    result = _classify_capacity_outcome(observed_oom=True, expect_oom=True)
    assert result == {
        "status": "expected_oom",
        "expectation": "cuda_oom",
        "expectation_met": True,
        "capacity_classification": "measured_single_device_capacity_failure",
        "single_device_oom_observed": True,
    }


def test_unexpected_completion_fails_closed() -> None:
    result = _classify_capacity_outcome(observed_oom=False, expect_oom=True)
    assert result["status"] == "unexpected_completion"
    assert result["expectation_met"] is False
    assert result["single_device_oom_observed"] is False


def test_unexpected_oom_fails_closed() -> None:
    result = _classify_capacity_outcome(observed_oom=True, expect_oom=False)
    assert result["status"] == "unexpected_oom"
    assert result["expectation_met"] is False
