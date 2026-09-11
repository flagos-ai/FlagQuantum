from __future__ import annotations

import pytest

from benchmarks.internal.evidence.general_mps_capacity import (
    PROFILES,
    TRUNCATION_BUDGET,
    capacity_baseline_payload,
    capacity_gradient_contract,
    cleanup_within_budget,
    logical_mps_bytes,
    reverse_checkpoint_capacity_bytes,
    validate_capacity_baseline,
)


@pytest.mark.unit
def test_a800_capacity_profile_exceeds_one_device_with_safe_eight_way_shards():
    n_sites, initial_bond, trained_bond = PROFILES["a800_capacity"]
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    assert logical_bytes > 80 << 30
    assert logical_bytes // 8 < 16 << 30
    assert trained_bond == initial_bond
    assert reverse_checkpoint_capacity_bytes(logical_bytes, 8) == (
        2 * (logical_bytes // 8) + (1 << 30)
    )
    assert capacity_gradient_contract("a800_capacity") == (
        "approximate",
        TRUNCATION_BUDGET,
    )


@pytest.mark.unit
def test_cleanup_budget_allows_runtime_residue_but_rejects_tensor_leaks():
    expected_limit = (40 << 30) // 1000
    assert cleanup_within_budget(8 << 20, 0, 40 << 30) == (True, expected_limit)
    assert cleanup_within_budget(42 << 20, 0, 40 << 30) == (
        False,
        expected_limit,
    )
    verified, limit = cleanup_within_budget(20 << 20, 8 << 20, 1 << 30)
    assert verified is False
    assert limit == (8 << 20) + ((1 << 30) // 1000)


def _records(world_size: int, status: str) -> list[dict]:
    return [
        {
            "rank": rank,
            "status": status,
            "peak_memory_bytes": 40 << 30,
        }
        for rank in range(world_size)
    ]


@pytest.mark.unit
def test_four_gpu_capacity_failure_can_be_paired_with_same_workload() -> None:
    n_sites = 12_288
    initial_bond = 512
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    payload = capacity_baseline_payload(
        world_size=4,
        status="cuda_oom",
        n_sites=n_sites,
        initial_bond=initial_bond,
        trained_bond=511,
        logical_bytes=logical_bytes,
        rank_records=_records(4, "cuda_oom"),
    )

    validate_capacity_baseline(
        payload,
        expected_world_size=4,
        n_sites=n_sites,
        initial_bond=initial_bond,
        logical_bytes=logical_bytes,
    )
    assert payload["capacity_failure"] is True
    assert payload["single_gpu_capacity_failure"] is False
    assert payload["scalability_claim_allowed"] is False


@pytest.mark.unit
def test_capacity_pairing_rejects_workload_or_world_size_mismatch() -> None:
    n_sites = 12_288
    initial_bond = 512
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    payload = capacity_baseline_payload(
        world_size=4,
        status="cuda_oom",
        n_sites=n_sites,
        initial_bond=initial_bond,
        trained_bond=511,
        logical_bytes=logical_bytes,
        rank_records=_records(4, "cuda_oom"),
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_capacity_baseline(
            payload,
            expected_world_size=1,
            n_sites=n_sites,
            initial_bond=initial_bond,
            logical_bytes=logical_bytes,
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_capacity_baseline(
            payload,
            expected_world_size=4,
            n_sites=n_sites + 1,
            initial_bond=initial_bond,
            logical_bytes=logical_bytes,
        )


@pytest.mark.unit
def test_capacity_baseline_rejects_non_capacity_rank_failures() -> None:
    n_sites = 12_288
    initial_bond = 512
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    records = _records(4, "cuda_oom")
    records[-1]["status"] = "runtime_error"
    payload = capacity_baseline_payload(
        world_size=4,
        status="runtime_error",
        n_sites=n_sites,
        initial_bond=initial_bond,
        trained_bond=511,
        logical_bytes=logical_bytes,
        rank_records=records,
    )

    assert payload["capacity_failure"] is False
    with pytest.raises(ValueError, match="capacity failure"):
        validate_capacity_baseline(
            payload,
            expected_world_size=4,
            n_sites=n_sites,
            initial_bond=initial_bond,
            logical_bytes=logical_bytes,
        )
