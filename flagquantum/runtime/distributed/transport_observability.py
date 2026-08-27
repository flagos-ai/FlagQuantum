"""Fail-closed contracts for observable FlagOS transport facts.

Profiler events can prove that a host transfer was observed inside a measured
collective interval.  Their absence cannot prove a device-direct route, and
neither profiler names nor successful collectives identify FlagCX.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

TRANSPORT_PROFILE_SCHEMA = "flagquantum_flagos_transport_observability_f6_v1"
TRANSPORT_RUN_SCHEMA = "flagquantum_flagos_transport_observability_run_v1"
TRANSPORT_RANK_SCHEMA = "flagquantum_flagos_transport_observability_rank_v1"
TRANSPORT_WORLD_SIZES = (2, 4, 8)
TRANSPORT_DTYPES = ("complex64", "complex128")
TRANSPORT_PRIMITIVES = (
    "all_gather_into_tensor",
    "all_reduce",
    "broadcast",
    "isend_irecv",
)

_HOST_TO_DEVICE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"memcpy\w*.*\b(host\s*(?:to|->)\s*device|h(?:to|2)d)\b",
        r"\b(host\s*(?:to|->)\s*device|h(?:to|2)d)\b.*memcpy\w*",
    )
)
_DEVICE_TO_HOST = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"memcpy\w*.*\b(device\s*(?:to|->)\s*host|d(?:to|2)h)\b",
        r"\b(device\s*(?:to|->)\s*host|d(?:to|2)h)\b.*memcpy\w*",
    )
)


class FlagOSTransportEvidenceError(ValueError):
    """Raised when F6 evidence is malformed, incomplete, or overclaimed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlagOSTransportEvidenceError(message)


def classify_explicit_host_transfer(event_name: str) -> str | None:
    """Classify only profiler names with an explicit host/device direction."""

    name = " ".join(str(event_name).split())
    if any(pattern.search(name) for pattern in _HOST_TO_DEVICE):
        return "host_to_device"
    if any(pattern.search(name) for pattern in _DEVICE_TO_HOST):
        return "device_to_host"
    return None


@dataclass(frozen=True)
class TransportProfilerEvent:
    """Aggregated profiler event retained as observation, not route identity."""

    name: str
    count: int
    self_cpu_time_us: float
    self_device_time_us: float
    host_transfer_direction: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "profiler event name is empty")
        _require(self.count >= 1, "profiler event count must be positive")
        _require(
            self.self_cpu_time_us >= 0 and self.self_device_time_us >= 0,
            "profiler event times must be non-negative",
        )
        expected = classify_explicit_host_transfer(self.name)
        _require(
            self.host_transfer_direction == expected,
            "host-transfer classification is not derived from the event name",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "self_cpu_time_us": self.self_cpu_time_us,
            "self_device_time_us": self.self_device_time_us,
            "host_transfer_direction": self.host_transfer_direction,
        }


@dataclass(frozen=True)
class FlagOSTransportObservation:
    """One rank-local collective observation and its bounded profiler summary."""

    primitive: str
    dtype: str
    passed: bool
    max_abs_error: float
    elapsed_seconds: float
    payload_bytes: int
    input_device_type: str
    output_device_type: str
    profiler_available: bool
    profiler_activities: tuple[str, ...]
    profiler_events: tuple[TransportProfilerEvent, ...]
    profiler_error: str | None = None

    def __post_init__(self) -> None:
        _require(self.primitive in TRANSPORT_PRIMITIVES, "unsupported primitive")
        _require(self.dtype in TRANSPORT_DTYPES, "unsupported dtype")
        _require(
            self.max_abs_error >= 0
            and self.elapsed_seconds >= 0
            and self.payload_bytes > 0,
            "invalid transport measurements",
        )
        _require(
            self.input_device_type == "flagos" and self.output_device_type == "flagos",
            "collective tensors escaped the FlagOS boundary",
        )
        if self.profiler_available:
            _require(
                self.profiler_error is None, "available profiler retained an error"
            )
            _require(bool(self.profiler_activities), "profiler activities are missing")
        else:
            _require(
                self.profiler_error is not None,
                "unavailable profiler requires an explicit error",
            )

    @property
    def explicit_host_transfer_events(self) -> tuple[TransportProfilerEvent, ...]:
        return tuple(
            event
            for event in self.profiler_events
            if event.host_transfer_direction is not None
        )

    @property
    def host_staging_observed(self) -> bool | None:
        return True if self.explicit_host_transfer_events else None

    @property
    def device_activity_observed(self) -> bool:
        return any(event.self_device_time_us > 0 for event in self.profiler_events)

    @property
    def profiler_capture_scope(self) -> str:
        return (
            "cpu_and_device" if self.device_activity_observed else "cpu_operator_only"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "dtype": self.dtype,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "elapsed_seconds": self.elapsed_seconds,
            "payload_bytes": self.payload_bytes,
            "input_device_type": self.input_device_type,
            "output_device_type": self.output_device_type,
            "profiler_available": self.profiler_available,
            "profiler_activities": self.profiler_activities,
            "profiler_events": [event.to_dict() for event in self.profiler_events],
            "explicit_host_transfer_events": [
                event.to_dict() for event in self.explicit_host_transfer_events
            ],
            "device_activity_observed": self.device_activity_observed,
            "profiler_capture_scope": self.profiler_capture_scope,
            "host_staging_observed": self.host_staging_observed,
            "profiler_error": self.profiler_error,
        }


def _validate_rank_record(
    record: Mapping[str, Any], *, world_size: int, rank: int
) -> bool:
    _require(record.get("schema") == TRANSPORT_RANK_SCHEMA, "invalid rank schema")
    _require(record.get("rank") == rank, "rank record mismatch")
    _require(record.get("local_rank") == rank, "local-rank record mismatch")
    _require(record.get("world_size") == world_size, "rank world size mismatch")
    _require(
        record.get("local_world_size") == world_size,
        "rank local world size mismatch",
    )
    _require(record.get("node_count") == 1, "F6 rank is not single-node")
    _require(record.get("device") == f"flagos:{rank}", "rank device mismatch")
    _require(record.get("device_type") == "flagos", "rank is not on FlagOS")
    observations = record.get("observations")
    _require(isinstance(observations, list), "rank observations are missing")
    required = {
        (primitive, dtype)
        for dtype in TRANSPORT_DTYPES
        for primitive in TRANSPORT_PRIMITIVES
    }
    observed = {
        (item.get("primitive"), item.get("dtype"))
        for item in observations
        if isinstance(item, dict)
    }
    _require(
        len(observations) == len(required) and observed == required,
        "rank transport matrix is incomplete",
    )
    for item in observations:
        _require(isinstance(item, dict), "transport observation is not an object")
        _require(
            item.get("passed") is True
            and item.get("input_device_type") == "flagos"
            and item.get("output_device_type") == "flagos"
            and item.get("payload_bytes", 0) > 0
            and item.get("profiler_available") is True
            and {"CPU", "CUDA"} <= set(item.get("profiler_activities", ()))
            and item.get("max_abs_error", float("inf"))
            <= (1e-5 if item.get("dtype") == "complex64" else 1e-12),
            "rank transport correctness or residency failed",
        )
        profiler_events = item.get("profiler_events")
        explicit_events = item.get("explicit_host_transfer_events")
        _require(
            isinstance(profiler_events, list) and isinstance(explicit_events, list),
            "profiler event lists are missing",
        )
        _require(bool(profiler_events), "profiler captured no operator events")
        derived_explicit = []
        for event in profiler_events:
            _require(isinstance(event, dict), "profiler event is not an object")
            direction = classify_explicit_host_transfer(str(event.get("name", "")))
            _require(
                event.get("host_transfer_direction") == direction,
                "profiler host-transfer direction was forged",
            )
            _require(
                isinstance(event.get("count"), int)
                and event["count"] >= 1
                and event.get("self_cpu_time_us", -1) >= 0
                and event.get("self_device_time_us", -1) >= 0,
                "profiler event measurements are invalid",
            )
            if direction is not None:
                derived_explicit.append(event)
        _require(
            explicit_events == derived_explicit,
            "explicit host-transfer events are not derived from the profiler",
        )
        _require(
            item.get("host_staging_observed") is (True if derived_explicit else None),
            "host-staging observation was inferred",
        )
        device_activity = any(
            event.get("self_device_time_us", 0) > 0 for event in profiler_events
        )
        _require(
            item.get("device_activity_observed") is device_activity,
            "device-activity status was not derived from profiler events",
        )
        _require(
            item.get("profiler_capture_scope")
            == ("cpu_and_device" if device_activity else "cpu_operator_only"),
            "profiler capture scope was not derived from events",
        )
    return True


@dataclass(frozen=True)
class FlagOSTransportObservabilityProfile:
    """A 2/4/8-card FlagOS transport observation ladder."""

    runs: tuple[Mapping[str, Any], ...]
    source_revision: str
    torch_fl_source_revision: str

    @property
    def accepted(self) -> bool:
        if len(self.runs) != len(TRANSPORT_WORLD_SIZES):
            return False
        try:
            for run, world_size in zip(self.runs, TRANSPORT_WORLD_SIZES):
                _require(
                    run.get("schema") == TRANSPORT_RUN_SCHEMA, "invalid run schema"
                )
                _require(run.get("status") == "passed", "run did not pass")
                _require(run.get("world_size") == world_size, "world-size ladder drift")
                _require(
                    run.get("local_world_size") == world_size,
                    "local world-size ladder drift",
                )
                _require(run.get("node_count") == 1, "F6 run is not single-node")
                _require(run.get("outer_backend") == "flagos", "not FlagOS")
                _require(
                    run.get("inner_communication_route") == "unattributed",
                    "inner route was inferred",
                )
                _require(run.get("flagcx_route_verified") is False, "FlagCX inferred")
                _require(
                    run.get("no_host_staging_certified") is False,
                    "absence of host staging was inferred",
                )
                _require(
                    run.get("communication_claim_allowed") is False,
                    "communication claim overreach",
                )
                _require(
                    run.get("scalability_claim_allowed") is False,
                    "scalability claim overreach",
                )
                _require(
                    run.get("production_support_claim_allowed") is False,
                    "production claim overreach",
                )
                _require(
                    run.get("release_gate_allowed") is False,
                    "release claim overreach",
                )
                ranks = run.get("ranks")
                _require(
                    isinstance(ranks, list) and len(ranks) == world_size,
                    "rank ladder incomplete",
                )
                for rank, record in enumerate(ranks):
                    _validate_rank_record(record, world_size=world_size, rank=rank)
                derived_transfers = [
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
                _require(
                    run.get("explicit_host_transfer_events") == derived_transfers,
                    "run host-transfer summary was not derived from ranks",
                )
                _require(
                    run.get("host_staging_observed")
                    is (True if derived_transfers else None),
                    "run host-staging status was inferred",
                )
            return True
        except FlagOSTransportEvidenceError:
            return False

    @property
    def explicit_host_transfer_events(self) -> tuple[Mapping[str, Any], ...]:
        events = []
        for run in self.runs:
            for rank in run.get("ranks", ()):
                for observation in rank.get("observations", ()):
                    for event in observation.get("explicit_host_transfer_events", ()):
                        events.append(
                            {
                                "world_size": run.get("world_size"),
                                "rank": rank.get("rank"),
                                "primitive": observation.get("primitive"),
                                "dtype": observation.get("dtype"),
                                **dict(event),
                            }
                        )
        return tuple(events)

    @property
    def host_staging_observed(self) -> bool | None:
        return True if self.explicit_host_transfer_events else None

    @property
    def device_activity_observed_all(self) -> bool:
        return all(
            observation.get("device_activity_observed") is True
            for run in self.runs
            for rank in run.get("ranks", ())
            for observation in rank.get("observations", ())
        )

    def require_accepted(self) -> None:
        if not self.accepted:
            raise FlagOSTransportEvidenceError("F6 transport observation gate failed")

    def to_dict(self) -> dict[str, Any]:
        accepted = self.accepted
        return {
            "schema": TRANSPORT_PROFILE_SCHEMA,
            "status": "passed" if accepted else "failed",
            "validation_scope": "flagos_single_node_transport_observability_ladder",
            "artifact_classification": "development_transport_observation",
            "world_sizes": list(TRANSPORT_WORLD_SIZES),
            "runs": [dict(run) for run in self.runs],
            "transport_observation_accepted": accepted,
            "outer_backend": "flagos",
            "inner_communication_route": "unattributed",
            "flagcx_route_verified": False,
            "explicit_host_transfer_events": [
                dict(event) for event in self.explicit_host_transfer_events
            ],
            "host_staging_observed": self.host_staging_observed,
            "no_host_staging_certified": False,
            "device_activity_observed_all": self.device_activity_observed_all,
            "profiler_capture_complete": self.device_activity_observed_all,
            "communication_claim_allowed": False,
            "scalability_claim_allowed": False,
            "production_support_claim_allowed": False,
            "release_gate_allowed": False,
            "environment": {
                "source_revision": self.source_revision,
                "torch_fl_source_revision": self.torch_fl_source_revision,
            },
            "blockers": [
                "provider_owned_inner_route_attestation_unavailable",
                "flagcx_route_unverified",
                (
                    "explicit_host_transfer_observed"
                    if self.host_staging_observed
                    else "absence_of_profiler_event_is_not_no_host_staging_proof"
                ),
                *(
                    ()
                    if self.device_activity_observed_all
                    else ("device_activity_not_observed_profiler_capture_incomplete",)
                ),
                "single_node_observation_not_performance_or_scalability_evidence",
            ],
        }


def build_flagos_transport_observability_profile(
    runs: Sequence[Mapping[str, Any]],
) -> FlagOSTransportObservabilityProfile:
    """Validate and aggregate a fixed 2/4/8-card F6 observation ladder."""

    ordered = tuple(
        sorted((dict(run) for run in runs), key=lambda run: run["world_size"])
    )
    _require(
        tuple(run.get("world_size") for run in ordered) == TRANSPORT_WORLD_SIZES,
        "F6 requires exactly the 2/4/8-card ladder",
    )
    revisions = {run.get("source_revision") for run in ordered}
    torch_fl_revisions = {run.get("torch_fl_source_revision") for run in ordered}
    _require(len(revisions) == 1 and None not in revisions, "source revisions differ")
    _require(
        len(torch_fl_revisions) == 1 and None not in torch_fl_revisions,
        "Torch-FL revisions differ",
    )
    profile = FlagOSTransportObservabilityProfile(
        runs=ordered,
        source_revision=str(revisions.pop()),
        torch_fl_source_revision=str(torch_fl_revisions.pop()),
    )
    profile.require_accepted()
    return profile


__all__ = (
    "FlagOSTransportEvidenceError",
    "FlagOSTransportObservation",
    "FlagOSTransportObservabilityProfile",
    "TRANSPORT_DTYPES",
    "TRANSPORT_PRIMITIVES",
    "TRANSPORT_PROFILE_SCHEMA",
    "TRANSPORT_RANK_SCHEMA",
    "TRANSPORT_RUN_SCHEMA",
    "TRANSPORT_WORLD_SIZES",
    "TransportProfilerEvent",
    "build_flagos_transport_observability_profile",
    "classify_explicit_host_transfer",
)
