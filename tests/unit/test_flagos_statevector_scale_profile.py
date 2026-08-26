from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from flagquantum.runtime.distributed.scale_profile import (
    FLAGOS_SCALE_CASES,
    FLAGOS_SCALE_DTYPES,
    FLAGOS_SCALE_WORLD_SIZES,
    FlagOSStatevectorScaleCase,
    FlagOSStatevectorScaleProfile,
    FlagOSStatevectorScaleRun,
    build_scale_profile,
    scale_run_from_dict,
)

pytestmark = pytest.mark.unit


def _case(name: str, dtype: str, world_size: int) -> FlagOSStatevectorScaleCase:
    total = 2**12
    local = total // world_size
    return FlagOSStatevectorScaleCase(
        name=name,
        dtype=dtype,
        passed=True,
        n_wires=12,
        local_amplitudes=local,
        total_amplitudes=total,
        local_state_bytes=local * (8 if dtype == "complex64" else 16),
        local_memory_bytes_by_rank=tuple(2_000_000 for _ in range(world_size)),
        memory_measurement="runtime_accounted_state_scratch_workspace",
        communication_count=3,
        communication_bytes=4096,
        peak_scratch_bytes=8192,
        distributed_gate_count=3,
        elapsed_seconds=0.2,
        max_abs_error=None if name == "capacity_invariant" else 0.0,
        norm_error=0.0,
        determinism_error=0.0,
        tolerance=2e-5 if dtype == "complex64" else 2e-11,
        persistent_wire_layout=name == "persistent_layout_reference",
        device_type="flagos",
        reference_scope=(
            "rank_local_invariant_only"
            if name == "capacity_invariant"
            else "bounded_per_rank_cpu_complex128_reference"
        ),
    )


def _run(world_size: int) -> FlagOSStatevectorScaleRun:
    return FlagOSStatevectorScaleRun(
        world_size=world_size,
        local_world_size=world_size,
        node_count=1,
        cases=tuple(
            _case(name, dtype, world_size)
            for dtype in FLAGOS_SCALE_DTYPES
            for name in FLAGOS_SCALE_CASES
        ),
        rank_placement=tuple(
            {
                "rank": rank,
                "local_rank": rank,
                "device_index": rank,
                "device": f"flagos:{rank}",
                "ownership": "distinct_amplitude_shard",
            }
            for rank in range(world_size)
        ),
        environment={"torch": "test", "torch_fl": "test"},
    )


def _profile() -> FlagOSStatevectorScaleProfile:
    return FlagOSStatevectorScaleProfile(
        runs=tuple(_run(world_size) for world_size in FLAGOS_SCALE_WORLD_SIZES),
        environment={"source_revision": "test"},
    )


def test_scale_profile_accepts_complete_2_4_8_ladder_but_no_stronger_claim():
    profile = _profile()
    profile.require_accepted()

    payload = profile.to_dict()
    assert payload["status"] == "passed"
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["scale_ladder_accepted"] is True
    assert payload["statevector_forward_scale_profile_accepted"] is True
    assert payload["flagcx_route_verified"] is False
    assert payload["host_staging_observed"] is None
    assert payload["communication_claim_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["production_support_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert "sharded_backward_and_optimizer_not_measured" in payload["blockers"]


def test_scale_profile_fails_closed_for_missing_world_size():
    profile = replace(_profile(), runs=(_run(2), _run(4)))

    assert profile.accepted is False
    assert "scale_ladder_incomplete" in profile.to_dict()["blockers"]
    with pytest.raises(RuntimeError, match="scale ladder failed"):
        profile.require_accepted()


def test_scale_run_rejects_duplicate_or_noncommunicating_cases():
    run = _run(2)
    assert replace(run, cases=run.cases + (run.cases[0],)).accepted is False
    failed = replace(run.cases[0], communication_count=0)
    assert replace(run, cases=(failed, *run.cases[1:])).accepted is False


def test_persistent_layout_accepts_layout_exchange_without_distributed_gate_count():
    run = _run(2)
    cases = tuple(
        replace(case, distributed_gate_count=0)
        if case.name == "persistent_layout_reference"
        else case
        for case in run.cases
    )

    assert replace(run, cases=cases).accepted is True


def test_scale_run_rejects_replicated_or_materialized_execution():
    run = _run(2)
    replicated = replace(run.cases[0], distribution_semantics="replicated_per_rank")
    assert replace(run, cases=(replicated, *run.cases[1:])).accepted is False
    materialized = replace(run.cases[0], full_state_materialization=True)
    assert replace(run, cases=(materialized, *run.cases[1:])).accepted is False


def test_scale_profile_round_trips_worker_payloads_and_sorts_world_sizes():
    runs = [_run(8).to_dict(), _run(2).to_dict(), _run(4).to_dict()]

    parsed = scale_run_from_dict(runs[0])
    profile = build_scale_profile(runs, environment={"source_revision": "test"})

    assert parsed.world_size == 8
    assert [run.world_size for run in profile.runs] == [2, 4, 8]
    assert profile.accepted is True


def test_scale_case_requires_per_rank_memory_ownership():
    with pytest.raises(ValueError, match="every rank"):
        replace(
            _case("capacity_invariant", "complex128", 2), local_memory_bytes_by_rank=()
        )


def test_scale_case_rejects_ambiguous_memory_measurement():
    with pytest.raises(ValueError, match="memory measurement"):
        replace(
            _case("capacity_invariant", "complex128", 2),
            memory_measurement="peak_memory",
        )


def test_checked_in_a800_scale_artifact_retains_fail_closed_boundary():
    artifact = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "flagos_statevector_scale_f2_a800_20260826.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["status"] == "passed"
    assert payload["world_sizes"] == [2, 4, 8]
    assert payload["scale_ladder_accepted"] is True
    assert payload["environment"]["source_revision"] == (
        "21e220960e9756882ece2932614f3cc4d368a49a"
    )
    assert all(run["scale_run_accepted"] for run in payload["runs"])
    assert all(
        case["memory_measurement"] == "runtime_accounted_state_scratch_workspace"
        for run in payload["runs"]
        for case in run["cases"]
    )
    assert payload["flagcx_route_verified"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
