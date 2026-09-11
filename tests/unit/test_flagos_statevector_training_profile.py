"""Fail-closed unit contracts for the F3 training profile."""

from dataclasses import replace

import pytest

from flagquantum.runtime.distributed.training_profile import (
    FLAGOS_TRAINING_CASES,
    FLAGOS_TRAINING_DTYPES,
    FlagOSTrainingCase,
    FlagOSTrainingRun,
    build_training_profile,
)

pytestmark = pytest.mark.unit


def _case(name: str, dtype: str) -> FlagOSTrainingCase:
    return FlagOSTrainingCase(
        name=name,
        dtype=dtype,
        passed=True,
        n_wires=6,
        steps=1 if name == "gradient_reference" else 3,
        local_amplitudes=32,
        total_amplitudes=64,
        value_max_abs_error=1e-12,
        gradient_max_abs_error=2e-12,
        parameter_max_abs_error=3e-12,
        rank_consistency_error=0.0,
        tolerance=3e-10,
        communication_count=4,
        communication_bytes=1024,
        peak_memory_bytes=4096,
        device_type="flagos",
    )


def _run(world_size: int) -> FlagOSTrainingRun:
    cases = tuple(
        _case(name, dtype)
        for dtype in FLAGOS_TRAINING_DTYPES
        for name in FLAGOS_TRAINING_CASES
    )
    return FlagOSTrainingRun(
        world_size=world_size,
        local_world_size=world_size,
        node_count=1,
        cases=cases,
        rank_placement=tuple(
            {"rank": rank, "device_index": rank} for rank in range(world_size)
        ),
        environment={"source_revision": "abc"},
    )


def test_complete_training_ladder_is_accepted_but_claims_remain_closed():
    profile = build_training_profile(
        [_run(8).to_dict(), _run(2), _run(4)], environment={"controller": "test"}
    )
    payload = profile.to_dict()
    assert profile.accepted
    profile.require_accepted()
    assert payload["world_sizes"] == [2, 4, 8]
    assert payload["sharded_backward_profile_accepted"] is True
    assert payload["sharded_optimizer_profile_accepted"] is True
    assert payload["gradient_distribution"] == "replicated_after_all_reduce"
    assert payload["optimizer_update_semantics"] == "owner_step_then_broadcast"
    assert payload["flagcx_route_verified"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("full_state_materialization", True),
        ("backward_uses_full_state_replay", True),
        ("gradient_distribution", "sharded_across_ranks"),
        ("optimizer_update_semantics", "replicated"),
        ("communication_count", 0),
        ("communication_bytes", 0),
        ("passed", False),
    ),
)
def test_training_run_rejects_missing_execution_evidence(field, value):
    run = _run(2)
    broken = replace(run.cases[0], **{field: value})
    assert replace(run, cases=(broken, *run.cases[1:])).accepted is False


def test_profile_fails_closed_when_world_size_is_missing():
    profile = build_training_profile([_run(2), _run(4)], environment={})
    assert profile.accepted is False
    assert "training_scale_ladder_incomplete" in profile.to_dict()["blockers"]
    with pytest.raises(RuntimeError, match="training ladder failed"):
        profile.require_accepted()


def test_run_rejects_flagcx_inference_and_duplicate_device_placement():
    with pytest.raises(ValueError, match="cannot infer"):
        replace(_run(2), flagcx_route_verified=True)
    run = _run(2)
    duplicated = replace(
        run,
        rank_placement=(
            {"rank": 0, "device_index": 0},
            {"rank": 1, "device_index": 0},
        ),
    )
    assert duplicated.accepted is False
