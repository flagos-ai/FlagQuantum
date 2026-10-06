"""Derive the claim boundaries a recorded multi-node run still leaves standing.

A blocker list written out by hand asserts a boundary the run never checked.
Everything here derives the same list from what the run recorded, so each
blocker traces to a field in the artifact and a later run that closes a gap
drops its blocker without anyone editing a literal.

Two kinds of boundary are kept apart, because only one of them can move.

*Observed* boundaries are computed from the artifact's own observations: which
transport NCCL actually selected, whether a repeated measurement was taken,
whether an interchange was profiled. These disappear when the observation says
so.

*Declared* boundaries are properties of the workload a probe was asked to run --
a two-node pair, one fixed circuit. No observation inside that run can retract
them, so a probe states them explicitly and `claim_blockers` only unions the two
sets. Presenting a declaration as a derivation would be exactly the failure this
module exists to prevent, so the two are never mixed.

Absence is not evidence. A profiler that observed no explicit host transfer has
not shown that none happened, so no derivation here retracts a blocker on a
missing observation: every retraction needs a positive one.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

# NCCL names the transport it selected, and it names the transport of every
# channel that carried traffic. Both are read, because a device selection
# without a channel line says which plugin loaded rather than which route moved
# the payload. The socket markers are not on their own a socket route: a fabric
# run logs the socket interface for its out-of-band bootstrap.
_NETWORK_SOCKET_MARKERS = ("NET/Socket : Using", "via NET/Socket/")
_NETWORK_FABRIC_MARKERS = ("NET/IB : Using", "via NET/IB/")
_NETWORK_FABRIC_ROCE_MARKER = "/RoCE"
_NETWORK_GPU_DIRECT_MARKER = "GDRDMA"

TRANSPORT_ROUTE_SOCKET = "socket"
TRANSPORT_ROUTE_FABRIC = "infiniband"
TRANSPORT_ROUTE_UNOBSERVED = "unobserved"
TRANSPORT_ROUTE_LEVELS = (
    TRANSPORT_ROUTE_SOCKET,
    TRANSPORT_ROUTE_FABRIC,
    TRANSPORT_ROUTE_UNOBSERVED,
)

CONFIGURED_TRANSPORT_UNSPECIFIED = "unspecified"

DEBUG_LOG_SCOPES = ("rank", "node")

# A blocker is retracted only by the positive observation it names. A number of
# iterations is not an observation on its own, so the floors below are the
# smallest record that still means what the blocker's absence claims: a
# measurement that warmed the path up and then repeated itself. One unsynchronized
# reading of a device kernel measures the launch, not the workload.
MINIMUM_WARMUP_ITERATIONS = 1
MINIMUM_MEASURED_ITERATIONS = 3

BLOCKER_RDMA_NOT_TESTED = "rdma_not_tested"
BLOCKER_PRODUCTION_PERFORMANCE_NOT_MEASURED = "production_performance_not_measured"
BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED = "hidden_host_staging_not_audited"
#: Staging that an audit found is a different statement from staging nobody
#: looked for, and a reader has to be able to tell them apart: the first is a
#: property of the workload, the second of the evidence. Both keep a blocker, so
#: neither reading retracts a limitation the earlier artifact claimed.
BLOCKER_HOST_STAGING_IN_MEASURED_REGION = "host_staging_in_measured_region"
BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER = (
    "validation_only_tiny_full_state_gather"
)
BLOCKER_VALIDATION_ONLY_TINY_FULL_MPS_GATHER = "validation_only_tiny_full_mps_gather"
BLOCKER_INTER_NODE_CUT_WIDTH_NOT_SWEPT = "inter_node_cut_width_not_swept"
BLOCKER_SLICE_COUNT_FIXED_AT_WORLD_SIZE = "slice_count_fixed_at_world_size"
BLOCKER_TWO_NODE_PAIR_ONLY = "two_node_pair_only_no_wider_topology"
BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY = "toy_circuit_parameters_only"

#: A sweep is at least two distinct widths. One width, however often it is run
#: and however much it exchanges, is the single point the blocker names.
MINIMUM_SWEPT_WIDTHS = 2

#: The smallest parameterization any frozen release workload in this repository
#: declares. It is derived from the contracts rather than chosen here: the
#: statevector release path's narrowest accepted matched-speed configuration is
#: 22 qubits at depth 8 (176 leaves), the tensor-network path's is 14 qubits at
#: six layers (84), and the MPS path's is 31 trainable leaves at every rung of
#: its matched-speed ladder, so 31 is the minimum. The floor is what keeps
#: `toy_circuit_claim_blockers` from being satisfied by a hand-written circuit
#: that agrees with a hand-written declaration; it is deliberately the loosest
#: value the three contracts allow, because a floor taken from one lane's
#: contract would make another lane's own frozen workload unreachable.
#: `tests/unit/test_claim_boundary.py::test_the_parameterization_floor_is_the_contracts_own_minimum`
#: recomputes the minimum from the three committed manifests, so a refrozen
#: contract that moves it fails the suite instead of silently rotting this
#: number.
MINIMUM_RELEASE_CIRCUIT_PARAMETERS = 31


class ClaimBoundaryError(ValueError):
    """An observation that cannot support the claim it is being recorded under."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any, *, floor: int) -> int | None:
    """A whole number at or above `floor`, or `None` when it is not one.

    `True` is rejected rather than read as 1, because a boolean is a flag about
    a count and not the count, and a JSON artefact that carried `true` where a
    number belongs is a producer defect to fail closed on rather than a value to
    arithmetic with.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= floor else None


def configured_infiniband_state(environment: Mapping[str, str]) -> bool | None:
    """What `NCCL_IB_DISABLE` asked for: refused, enabled, or never stated.

    The three answers are kept apart because they are three different runs. An
    unset variable leaves the decision to NCCL, so a lane that reported it as
    "enabled" would be claiming a configuration it never made.
    """

    value = environment.get("NCCL_IB_DISABLE")
    if value is None or not value.strip():
        return None
    return value.strip() == "1"


def observed_network_route(
    log: str | None,
    *,
    configured_interface: str | None,
    infiniband_disabled: bool | None,
    debug_log_scope: str,
) -> dict[str, Any]:
    """What the NCCL debug log proves about the route this run took.

    The route is read off the log rather than inferred from the environment: a
    configured value says what was asked for, and the whole point of recording
    it is that NCCL does not fail when it chooses something else. The two are
    cross-checked, so a run configured for one transport and observed on the
    other is refused instead of being recorded as the one that was asked for.

    `infiniband_disabled` is `None` when the run stated no preference, which is
    not the same as stating that the fabric may be used. With no log there is
    nothing to check, and the answer is `unobserved` at
    `configured_only` evidence level.

    Raises:
        ClaimBoundaryError: the log names no route, or names one other than the
            configured transport, or omits the configured interface.
    """

    if debug_log_scope not in DEBUG_LOG_SCOPES:
        raise ClaimBoundaryError(
            f"debug log scope {debug_log_scope!r} is neither of {DEBUG_LOG_SCOPES}"
        )
    if infiniband_disabled is None:
        configured_transport = CONFIGURED_TRANSPORT_UNSPECIFIED
    elif infiniband_disabled:
        configured_transport = TRANSPORT_ROUTE_SOCKET
    else:
        configured_transport = TRANSPORT_ROUTE_FABRIC

    observation: dict[str, Any] = {
        "backend": "nccl",
        "configured_interface": configured_interface,
        "configured_transport": configured_transport,
        "route": TRANSPORT_ROUTE_UNOBSERVED,
        "evidence_level": "configured_only",
        # One `torchrun` per node writes one debug log, so above one rank per
        # node the file holds every local rank's view rather than one rank's.
        # Recorded rather than assumed: the route reads the same either way,
        # but how much of the traffic it covers does not.
        "debug_log_scope": debug_log_scope,
    }
    if log is None:
        return observation

    fabric_observed = any(marker in log for marker in _NETWORK_FABRIC_MARKERS)
    observation.update(
        {
            "debug_log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
            "socket_transport_observed": any(
                marker in log for marker in _NETWORK_SOCKET_MARKERS
            ),
            "infiniband_transport_observed": fabric_observed,
            "roce_transport_observed": fabric_observed
            and _NETWORK_FABRIC_ROCE_MARKER in log,
            "gpu_direct_observed": fabric_observed
            and _NETWORK_GPU_DIRECT_MARKER in log,
            "configured_interface_observed": bool(
                configured_interface and configured_interface in log
            ),
            "evidence_level": "observed_debug_log",
        }
    )
    if fabric_observed:
        route = TRANSPORT_ROUTE_FABRIC
    elif observation["socket_transport_observed"]:
        route = TRANSPORT_ROUTE_SOCKET
    else:
        raise ClaimBoundaryError(
            "the NCCL debug log names neither a socket transport nor an "
            "InfiniBand one, so the route this run used was not observed"
        )
    if (
        configured_transport != CONFIGURED_TRANSPORT_UNSPECIFIED
        and route != configured_transport
    ):
        raise ClaimBoundaryError(
            f"the NCCL debug log shows the {route} route, which does not confirm "
            f"the configured {configured_transport} transport"
        )
    if (
        configured_interface is not None
        and not observation["configured_interface_observed"]
    ):
        raise ClaimBoundaryError(
            f"the NCCL debug log does not confirm the configured interface "
            f"{configured_interface}"
        )
    observation["route"] = route
    return observation


def measured_performance(
    seconds: Sequence[float],
    *,
    warmup_iterations: int,
    measurement: str,
) -> dict[str, Any]:
    """The record `measurement_claim_blockers` accepts, built from raw samples.

    Built here rather than in each probe so that the shape a probe records and
    the shape the derivation reads cannot drift apart.

    Raises:
        ClaimBoundaryError: too few samples, or no warmup.
    """

    values = [float(value) for value in seconds]
    if len(values) < MINIMUM_MEASURED_ITERATIONS:
        raise ClaimBoundaryError(
            f"a measurement needs at least {MINIMUM_MEASURED_ITERATIONS} samples, "
            f"received {len(values)}"
        )
    if warmup_iterations < MINIMUM_WARMUP_ITERATIONS:
        raise ClaimBoundaryError(
            f"a measurement needs at least {MINIMUM_WARMUP_ITERATIONS} warmup "
            f"iteration, received {warmup_iterations}"
        )
    if not measurement.strip():
        raise ClaimBoundaryError("a measurement has to name what it measured")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ClaimBoundaryError("measured durations must be finite and positive")
    ordered = sorted(values)
    return {
        "measured": True,
        # Every sample is taken across a device synchronization, so the number
        # is the workload's duration rather than the launch's.
        "synchronized": True,
        "measurement": measurement,
        "warmup_iterations": int(warmup_iterations),
        "measured_iterations": len(values),
        "seconds": values,
        "minimum_seconds": ordered[0],
        "median_seconds": _median(ordered),
        "maximum_seconds": ordered[-1],
    }


def _median(ordered: Sequence[float]) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _measured_seconds(observations: Mapping[str, Any]) -> tuple[float, ...] | None:
    performance = _mapping(observations.get("performance"))
    if performance.get("measured") is not True:
        return None
    if performance.get("synchronized") is not True:
        return None
    if (
        _count(performance.get("warmup_iterations"), floor=MINIMUM_WARMUP_ITERATIONS)
        is None
    ):
        return None
    iterations = _count(
        performance.get("measured_iterations"), floor=MINIMUM_MEASURED_ITERATIONS
    )
    if iterations is None:
        return None
    samples = performance.get("seconds")
    if not isinstance(samples, list) or len(samples) != iterations:
        return None
    try:
        values = tuple(float(sample) for sample in samples)
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        return None
    return values


def transport_claim_blockers(observations: Mapping[str, Any]) -> tuple[str, ...]:
    """`rdma_not_tested` stands until the log shows a fabric route was used."""

    network = _mapping(observations.get("network"))
    if str(network.get("route")) == TRANSPORT_ROUTE_FABRIC:
        return ()
    return (BLOCKER_RDMA_NOT_TESTED,)


def measurement_claim_blockers(observations: Mapping[str, Any]) -> tuple[str, ...]:
    """`production_performance_not_measured` stands until a measurement exists.

    This is a positive existence claim and not a claim about scale: a workload
    that is too small to be a production one says so through
    `toy_circuit_parameters_only`, which `toy_circuit_claim_blockers` retires
    only on the parameterization the circuit itself bound.
    """

    if _measured_seconds(observations) is None:
        return (BLOCKER_PRODUCTION_PERFORMANCE_NOT_MEASURED,)
    return ()


def gather_claim_blockers(
    *, production_materialization: bool, blocker: str
) -> tuple[str, ...]:
    """A full-state gather taken to check an answer is not a capacity result.

    The gathered tensor proves the answer; it is not the distribution the claim
    would be about. Until a materialization that the workload itself needs is
    recorded, the blocker stands however small the gathered state is.
    """

    return () if production_materialization else (blocker,)


def cut_width_claim_blockers(
    observed_widths: Mapping[Any, Any],
    *,
    blocker: str = BLOCKER_INTER_NODE_CUT_WIDTH_NOT_SWEPT,
) -> tuple[str, ...]:
    """`inter_node_cut_width_not_swept` stands until a cut moved and stayed cut.

    Two numbers are not a sweep. A width only counts when the exchange it
    implies still crossed the hosts there, which is why the caller passes the
    widths it exercised together with the inter-node bytes recorded at each. A
    width that exchanged nothing across the hosts is a locally reconstructed
    state, and counting it would record a wider run that never crossed the
    boundary the blocker is about.
    """

    crossed = {
        int(width)
        for width, inter_node_bytes in observed_widths.items()
        if inter_node_bytes
    }
    if len(crossed) >= MINIMUM_SWEPT_WIDTHS:
        return ()
    return (blocker,)


def slice_count_claim_blockers(
    observed_counts: Mapping[Any, Any],
) -> tuple[str, ...]:
    """`slice_count_fixed_at_world_size` stands until the slice count varied.

    The counts carry the same condition as a cut width: a slice count whose
    ranks exchanged nothing has not shown the workload partitioned differently,
    only that it ran at a different size.
    """

    varied = {
        int(count)
        for count, inter_node_bytes in observed_counts.items()
        if inter_node_bytes
    }
    if len(varied) >= MINIMUM_SWEPT_WIDTHS:
        return ()
    return (BLOCKER_SLICE_COUNT_FIXED_AT_WORLD_SIZE,)


def staging_claim_blockers(observations: Mapping[str, Any]) -> tuple[str, ...]:
    """Which host-staging blocker the audit supports, if either.

    A profiler that was unavailable has shown nothing, so the blocker keeps the
    name it had when nobody had looked. An audit that ran and saw an explicit
    host transfer inside the profiled region reports that transfer under its own
    name, because "nobody looked" would then be false: the audit looked, and
    what it found is a property of the workload. Only an audit that ran and saw
    no host transfer clears the group.
    """

    staging = _mapping(observations.get("host_staging"))
    observed = staging.get("host_transfer_observed")
    if staging.get("profiled") is not True:
        return (BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED,)
    if observed is True:
        return (BLOCKER_HOST_STAGING_IN_MEASURED_REGION,)
    if observed is not False:
        # A profiler that ran and recorded no verdict is an incomplete audit,
        # which is not the same as an audit that saw nothing.
        return (BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED,)
    return ()


def toy_circuit_claim_blockers(observations: Mapping[str, Any]) -> tuple[str, ...]:
    """`toy_circuit_parameters_only` stands until the circuit bound real leaves.

    The blocker is a statement about the parameterization, so the observation
    that retires it is the parameterization: the frozen configuration the
    workload ran, the leaf count the release contract declares for it, and the
    leaf count the built circuit reported. A probe that ran the contract's own
    workload binds the declared count at the declared width; a probe that ran a
    hand-written few-leaf circuit binds neither and keeps the blocker.

    The floor is the project's own smallest accepted released circuit rather than
    a number chosen here, so a small circuit cannot retire the blocker by naming
    a small manifest. The one thing this cannot see through is a probe that
    builds a large circuit and reports a declared count that does not match it,
    which is why the artifact also records the manifest it read and the count it
    bound: the declaration is checkable against the file it names.
    """

    circuit = _mapping(observations.get("frozen_circuit"))
    configuration = circuit.get("configuration")
    digest = circuit.get("manifest_sha256")
    declared = _count(
        circuit.get("declared_parameter_count"),
        floor=MINIMUM_RELEASE_CIRCUIT_PARAMETERS,
    )
    bound = _count(
        circuit.get("bound_parameter_count"),
        floor=MINIMUM_RELEASE_CIRCUIT_PARAMETERS,
    )
    if not isinstance(configuration, str) or not configuration:
        return (BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,)
    if not isinstance(digest, str) or len(digest) != 64:
        return (BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,)
    if declared is None or bound is None or declared != bound:
        return (BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY,)
    return ()


def claim_blockers(*groups: Sequence[str]) -> list[str]:
    """One sorted, de-duplicated blocker list from every group that applies."""

    return sorted({str(blocker) for group in groups for blocker in group if blocker})


__all__ = (
    "BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED",
    "BLOCKER_HOST_STAGING_IN_MEASURED_REGION",
    "BLOCKER_INTER_NODE_CUT_WIDTH_NOT_SWEPT",
    "BLOCKER_PRODUCTION_PERFORMANCE_NOT_MEASURED",
    "BLOCKER_RDMA_NOT_TESTED",
    "BLOCKER_SLICE_COUNT_FIXED_AT_WORLD_SIZE",
    "BLOCKER_TOY_CIRCUIT_PARAMETERS_ONLY",
    "BLOCKER_TWO_NODE_PAIR_ONLY",
    "BLOCKER_VALIDATION_ONLY_TINY_FULL_MPS_GATHER",
    "BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER",
    "CONFIGURED_TRANSPORT_UNSPECIFIED",
    "DEBUG_LOG_SCOPES",
    "MINIMUM_MEASURED_ITERATIONS",
    "MINIMUM_RELEASE_CIRCUIT_PARAMETERS",
    "MINIMUM_SWEPT_WIDTHS",
    "MINIMUM_WARMUP_ITERATIONS",
    "TRANSPORT_ROUTE_FABRIC",
    "TRANSPORT_ROUTE_LEVELS",
    "TRANSPORT_ROUTE_SOCKET",
    "TRANSPORT_ROUTE_UNOBSERVED",
    "ClaimBoundaryError",
    "claim_blockers",
    "configured_infiniband_state",
    "cut_width_claim_blockers",
    "gather_claim_blockers",
    "measured_performance",
    "measurement_claim_blockers",
    "observed_network_route",
    "slice_count_claim_blockers",
    "staging_claim_blockers",
    "toy_circuit_claim_blockers",
    "transport_claim_blockers",
)
