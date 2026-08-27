"""Fail-closed unit contracts for FlagOS F6 transport observations."""

from __future__ import annotations

import copy

import pytest

from flagquantum.runtime.distributed.transport_observability import (
    TRANSPORT_DTYPES,
    TRANSPORT_PRIMITIVES,
    TRANSPORT_RANK_SCHEMA,
    TRANSPORT_RUN_SCHEMA,
    FlagOSTransportEvidenceError,
    FlagOSTransportObservation,
    TransportProfilerEvent,
    build_flagos_transport_observability_profile,
    classify_explicit_host_transfer,
)

pytestmark = [pytest.mark.unit, pytest.mark.distributed_cpu]


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("Memcpy HtoD (Pageable -> Device)", "host_to_device"),
        ("cudaMemcpyAsync D2H", "device_to_host"),
        ("Memcpy Device -> Host", "device_to_host"),
        ("aten::copy_", None),
        ("cudaMemcpyAsync", None),
        ("flagcx::all_reduce", None),
    ),
)
def test_host_transfer_classifier_requires_explicit_direction(name, expected):
    assert classify_explicit_host_transfer(name) == expected


def _observation(
    primitive: str, dtype: str, *, transfer: str | None = None
) -> dict[str, object]:
    events = (
        TransportProfilerEvent(
            name="c10d::collective",
            count=1,
            self_cpu_time_us=2.0,
            self_device_time_us=3.0,
        ),
    )
    if transfer is not None:
        events += (
            TransportProfilerEvent(
                name=transfer,
                count=1,
                self_cpu_time_us=1.0,
                self_device_time_us=1.0,
                host_transfer_direction=classify_explicit_host_transfer(transfer),
            ),
        )
    return FlagOSTransportObservation(
        primitive=primitive,
        dtype=dtype,
        passed=True,
        max_abs_error=0.0,
        elapsed_seconds=0.01,
        payload_bytes=16,
        input_device_type="flagos",
        output_device_type="flagos",
        profiler_available=True,
        profiler_activities=("CPU", "CUDA"),
        profiler_events=events,
    ).to_dict()


def _rank(rank: int, world_size: int, *, transfer: str | None = None):
    return {
        "schema": TRANSPORT_RANK_SCHEMA,
        "rank": rank,
        "local_rank": rank,
        "world_size": world_size,
        "local_world_size": world_size,
        "node_count": 1,
        "device": f"flagos:{rank}",
        "device_type": "flagos",
        "physical_device_name": "development-gpu",
        "observations": [
            _observation(
                primitive,
                dtype,
                transfer=(
                    transfer
                    if rank == 0 and dtype == "complex128" and primitive == "all_reduce"
                    else None
                ),
            )
            for dtype in TRANSPORT_DTYPES
            for primitive in TRANSPORT_PRIMITIVES
        ],
        "environment": {"torch": "test", "torch_fl": "test"},
    }


def _runs(*, transfer: str | None = None):
    runs = []
    for world_size in (2, 4, 8):
        ranks = [
            _rank(rank, world_size, transfer=transfer) for rank in range(world_size)
        ]
        explicit_transfers = [
            {
                "rank": rank["rank"],
                "primitive": observation["primitive"],
                "dtype": observation["dtype"],
                **event,
            }
            for rank in ranks
            for observation in rank["observations"]
            for event in observation["explicit_host_transfer_events"]
        ]
        runs.append(
            {
                "schema": TRANSPORT_RUN_SCHEMA,
                "status": "passed",
                "world_size": world_size,
                "local_world_size": world_size,
                "node_count": 1,
                "outer_backend": "flagos",
                "inner_communication_route": "unattributed",
                "flagcx_route_verified": False,
                "host_staging_observed": True if explicit_transfers else None,
                "no_host_staging_certified": False,
                "communication_claim_allowed": False,
                "scalability_claim_allowed": False,
                "production_support_claim_allowed": False,
                "release_gate_allowed": False,
                "ranks": ranks,
                "explicit_host_transfer_events": explicit_transfers,
                "source_revision": "a" * 40,
                "torch_fl_source_revision": "b" * 40,
            }
        )
    return runs


def test_profile_accepts_observations_without_claiming_route_or_no_staging():
    payload = build_flagos_transport_observability_profile(_runs()).to_dict()

    assert payload["transport_observation_accepted"] is True
    assert payload["inner_communication_route"] == "unattributed"
    assert payload["flagcx_route_verified"] is False
    assert payload["host_staging_observed"] is None
    assert payload["no_host_staging_certified"] is False
    assert payload["device_activity_observed_all"] is True
    assert payload["profiler_capture_complete"] is True
    assert payload["communication_claim_allowed"] is False


def test_profile_retains_cpu_only_capture_as_an_explicit_blocker():
    runs = copy.deepcopy(_runs())
    for run in runs:
        for rank in run["ranks"]:
            for observation in rank["observations"]:
                for event in observation["profiler_events"]:
                    event["self_device_time_us"] = 0.0
                observation["device_activity_observed"] = False
                observation["profiler_capture_scope"] = "cpu_operator_only"

    payload = build_flagos_transport_observability_profile(runs).to_dict()

    assert payload["transport_observation_accepted"] is True
    assert payload["device_activity_observed_all"] is False
    assert payload["profiler_capture_complete"] is False
    assert (
        "device_activity_not_observed_profiler_capture_incomplete"
        in payload["blockers"]
    )
    assert payload["host_staging_observed"] is None
    assert payload["no_host_staging_certified"] is False
    assert (
        "absence_of_profiler_event_is_not_no_host_staging_proof" in payload["blockers"]
    )


def test_profile_promotes_only_explicit_transfer_presence_not_route_identity():
    payload = build_flagos_transport_observability_profile(
        _runs(transfer="Memcpy DtoH")
    ).to_dict()

    assert payload["host_staging_observed"] is True
    assert payload["explicit_host_transfer_events"]
    assert payload["flagcx_route_verified"] is False
    assert payload["communication_claim_allowed"] is False


@pytest.mark.parametrize(
    "mutate",
    (
        lambda runs: runs[0].__setitem__("inner_communication_route", "flagcx"),
        lambda runs: runs[0].__setitem__("status", "failed"),
        lambda runs: runs[1].__setitem__("flagcx_route_verified", True),
        lambda runs: runs[2].__setitem__("no_host_staging_certified", True),
        lambda runs: runs[1].__setitem__("scalability_claim_allowed", True),
        lambda runs: runs[0]["ranks"].pop(),
        lambda runs: runs[0]["ranks"][0].__setitem__("device", "flagos:1"),
        lambda runs: runs[1]["ranks"][0]["observations"][0].__setitem__(
            "profiler_available", False
        ),
        lambda runs: runs[2]["ranks"][0]["observations"][0].__setitem__(
            "output_device_type", "cpu"
        ),
        lambda runs: runs[2]["ranks"][0]["observations"][0].__setitem__(
            "device_activity_observed",
            not runs[2]["ranks"][0]["observations"][0]["device_activity_observed"],
        ),
        lambda runs: runs[1]["ranks"][0]["observations"][0].__setitem__(
            "profiler_events", []
        ),
        lambda runs: runs[0]["ranks"][0]["observations"][0]["profiler_events"][
            0
        ].__setitem__("host_transfer_direction", "device_to_host"),
    ),
)
def test_profile_rejects_incomplete_unprofiled_or_overclaimed_evidence(mutate):
    runs = copy.deepcopy(_runs())
    mutate(runs)
    with pytest.raises(FlagOSTransportEvidenceError):
        build_flagos_transport_observability_profile(runs)
