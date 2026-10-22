from __future__ import annotations

import pytest

from benchmarks.internal.evidence.general_mps_capacity import (
    CAPACITY_BASELINE_WORLDS,
    PROFILES,
    TARGET_WORLD,
    TRUNCATION_BUDGET,
    capacity_baseline_payload,
    capacity_baseline_worlds,
    capacity_gate_failure_worlds,
    capacity_gradient_contract,
    cleanup_within_budget,
    logical_mps_bytes,
    reverse_checkpoint_capacity_bytes,
    target_boundaries,
    topology_fingerprint,
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


@pytest.mark.unit
def test_rank_boundaries_follow_the_frozen_geometry_and_nest() -> None:
    """A capacity series needs each rank count to cut the same chain.

    Nesting is what makes the series one measurement: every cut a narrower world
    makes is also a cut a wider one makes, so the points differ in how many
    pieces the chain is in and not in where it is divided.
    """
    n_sites = PROFILES["capacity"][0]

    assert target_boundaries(n_sites, 1) == ()
    for world in (2, 4, 8, 16):
        assert len(target_boundaries(n_sites, world)) == world - 1

    cuts = {world: set(target_boundaries(n_sites, world)) for world in (2, 4, 8, 16)}
    assert cuts[2] <= cuts[4] <= cuts[8] <= cuts[16]

    # The default is the geometry the frozen eight-rank workload was measured at,
    # so an unparameterized call must still build exactly that circuit.
    assert target_boundaries(n_sites) == target_boundaries(n_sites, TARGET_WORLD)
    assert len(target_boundaries(n_sites)) == 7


@pytest.mark.unit
def test_geometry_is_workload_identity_not_deployment() -> None:
    """Two runs laid out for different slice counts are different workloads.

    The frozen geometry decides which gates exist, so it travels into the
    fingerprint. Without that, a baseline measured at one layout could be
    divided into a completion measured at another and the ratio would describe
    two different circuits.
    """
    n_sites, initial_bond = PROFILES["capacity"][0], PROFILES["capacity"][1]

    assert topology_fingerprint(n_sites, initial_bond, 8) != topology_fingerprint(
        n_sites, initial_bond, 4
    )
    assert topology_fingerprint(n_sites, initial_bond) == topology_fingerprint(
        n_sites, initial_bond, TARGET_WORLD
    )


@pytest.mark.unit
def test_a_series_point_pairs_only_within_its_own_geometry() -> None:
    """A baseline from one geometry must not validate against another."""
    n_sites, initial_bond = 12_288, 512
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    payload = capacity_baseline_payload(
        world_size=2,
        status="cuda_oom",
        n_sites=n_sites,
        initial_bond=initial_bond,
        trained_bond=511,
        logical_bytes=logical_bytes,
        rank_records=_records(2, "cuda_oom"),
        target_world=16,
    )

    assert payload["target_world_size"] == 16
    validate_capacity_baseline(
        payload,
        expected_world_size=2,
        n_sites=n_sites,
        initial_bond=initial_bond,
        logical_bytes=logical_bytes,
        target_world=16,
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_capacity_baseline(
            payload,
            expected_world_size=2,
            n_sites=n_sites,
            initial_bond=initial_bond,
            logical_bytes=logical_bytes,
            target_world=8,
        )


@pytest.mark.unit
def test_series_points_are_emittable_at_every_power_of_two_below_completion() -> None:
    assert capacity_baseline_worlds(8) == (1, 2, 4)
    assert capacity_baseline_worlds(16) == (1, 2, 4, 8)
    assert capacity_baseline_worlds(2) == (1,)
    assert capacity_baseline_worlds(TARGET_WORLD) == CAPACITY_BASELINE_WORLDS

    n_sites, initial_bond = 12_288, 512
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    for world in capacity_baseline_worlds(16):
        payload = capacity_baseline_payload(
            world_size=world,
            status="cuda_oom",
            n_sites=n_sites,
            initial_bond=initial_bond,
            trained_bond=511,
            logical_bytes=logical_bytes,
            rank_records=_records(world, "cuda_oom"),
            target_world=16,
        )
        assert payload["world_size"] == world
        assert payload["capacity_failure"] is True
        # No series point may promote a claim on its own.
        assert payload["scalability_claim_allowed"] is False
        assert payload["release_gate_allowed"] is False

    with pytest.raises(ValueError, match="baseline world size"):
        capacity_baseline_payload(
            world_size=16,
            status="cuda_oom",
            n_sites=n_sites,
            initial_bond=initial_bond,
            trained_bond=511,
            logical_bytes=logical_bytes,
            rank_records=_records(16, "cuda_oom"),
            target_world=16,
        )


@pytest.mark.unit
def test_adding_series_points_does_not_add_gate_conditions() -> None:
    """The gate still asks for one rank and for half the completion's ranks.

    A series measures intermediate rank counts to show how capacity moves. If
    those points became required failures, sealing a richer series would
    retroactively block a completion that already paired what the gate names.
    """
    assert capacity_gate_failure_worlds(8) == (1, 4)
    assert capacity_gate_failure_worlds(16) == (1, 8)
    assert capacity_gate_failure_worlds(4) == (1, 2)
    assert capacity_gate_failure_worlds(1) == (1,)

    for world in (4, 8, 16):
        assert set(capacity_gate_failure_worlds(world)) <= set(
            capacity_baseline_worlds(world)
        )
