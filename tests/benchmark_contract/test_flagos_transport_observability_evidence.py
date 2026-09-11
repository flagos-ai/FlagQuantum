"""Fail-closed checks for the checked-in FlagOS F6 transport evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flagquantum.runtime.distributed.transport_observability import (
    TRANSPORT_DTYPES,
    TRANSPORT_PRIMITIVES,
    TRANSPORT_WORLD_SIZES,
    build_flagos_transport_observability_profile,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.distributed_cpu]

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/flagos_transport_observability_f6_a800_20260827.json"


def test_checked_in_f6_artifact_rebuilds_and_keeps_transport_claims_closed():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    profile = build_flagos_transport_observability_profile(payload["runs"])
    profile.require_accepted()

    assert profile.to_dict() == payload
    assert payload["transport_observation_accepted"] is True
    assert payload["inner_communication_route"] == "unattributed"
    assert payload["flagcx_route_verified"] is False
    assert payload["host_staging_observed"] is None
    assert payload["no_host_staging_certified"] is False
    assert payload["device_activity_observed_all"] is False
    assert payload["profiler_capture_complete"] is False
    assert payload["communication_claim_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["production_support_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False


def test_checked_in_f6_artifact_preserves_the_measured_matrix_and_cupti_boundary():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    runs = payload["runs"]
    observations = [
        observation
        for run in runs
        for rank in run["ranks"]
        for observation in rank["observations"]
    ]

    assert tuple(run["world_size"] for run in runs) == TRANSPORT_WORLD_SIZES
    assert all(len(run["ranks"]) == run["world_size"] for run in runs)
    assert len(observations) == sum(TRANSPORT_WORLD_SIZES) * len(
        TRANSPORT_DTYPES
    ) * len(TRANSPORT_PRIMITIVES)
    assert {
        (observation["primitive"], observation["dtype"]) for observation in observations
    } == {
        (primitive, dtype)
        for dtype in TRANSPORT_DTYPES
        for primitive in TRANSPORT_PRIMITIVES
    }
    assert all(
        observation["passed"] is True
        and observation["max_abs_error"] == 0.0
        and observation["input_device_type"] == "flagos"
        and observation["output_device_type"] == "flagos"
        and observation["profiler_capture_scope"] == "cpu_operator_only"
        and observation["device_activity_observed"] is False
        and [event["name"] for event in observation["profiler_events"]]
        == ["aten::view_as_real"]
        for observation in observations
    )
    assert (
        "device_activity_not_observed_profiler_capture_incomplete"
        in payload["blockers"]
    )
    assert (
        "absence_of_profiler_event_is_not_no_host_staging_proof" in payload["blockers"]
    )
