"""Contracts for the observed-versus-declared boundary of a distributed claim.

A blocker in a two-node artifact is a statement about what a run did not show.
Deriving one from the run's own observations is what keeps it honest in both
directions: it cannot be retracted by editing a literal, and it cannot be
retracted by absence either. These tests fail if a helper clears a blocker
without the positive observation that would justify it, or if a route, a
measurement or a sweep is read more generously than the evidence allows.
"""

from __future__ import annotations

import math

import pytest

from flagquantum.runtime.audit.claim_boundary import (
    BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED,
    BLOCKER_HOST_STAGING_IN_MEASURED_REGION,
    BLOCKER_INTER_NODE_CUT_WIDTH_NOT_SWEPT,
    BLOCKER_PRODUCTION_PERFORMANCE_NOT_MEASURED,
    BLOCKER_RDMA_NOT_TESTED,
    BLOCKER_SLICE_COUNT_FIXED_AT_WORLD_SIZE,
    BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,
    ClaimBoundaryError,
    claim_blockers,
    configured_infiniband_state,
    cut_width_claim_blockers,
    gather_claim_blockers,
    measured_performance,
    measurement_claim_blockers,
    observed_network_route,
    slice_count_claim_blockers,
    staging_claim_blockers,
    transport_claim_blockers,
)

pytestmark = pytest.mark.unit

_SOCKET_LOG = """\
host: rank 0
NET/Socket : Using [0]ens22f0:10.1.15.172<0>
via NET/Socket/0
"""

_ROCE_LOG = """\
host: rank 0
NET/IB : Using [0]mlx5_101:1/RoCE [RO]; OOB ens22f0:10.1.15.172<0>
via NET/IB/0/GDRDMA
NET/GIN_IB_GDAKI : GPU Direct RDMA Enabled
"""

_BOTH_ROUTES_LOG = _ROCE_LOG + "NET/Socket : Using [0]ens22f0:10.1.15.172<0>\n"


def _route(
    log: str | None, *, interface: str = "ens22f0", transport: str = "fabric"
) -> dict:
    """Read a route from a log, as a run configured for `transport` would."""

    disabled = {"fabric": False, "socket": True, "unspecified": None}[transport]
    return observed_network_route(
        log,
        configured_interface=interface,
        infiniband_disabled=disabled,
        debug_log_scope="rank",
    )


def test_an_unread_log_reports_a_configured_route_and_not_an_observed_one() -> None:
    """Without a log there is nothing to observe, however the run was launched."""

    unobserved = _route(None)

    assert unobserved["evidence_level"] == "configured_only"
    assert unobserved["route"] == "unobserved"
    # The configuration is reported, but it is reported as configuration: a
    # reader has to be able to tell the two apart without knowing the schema.
    assert unobserved["configured_interface"] == "ens22f0"
    assert unobserved["configured_transport"] == "infiniband"
    assert "debug_log_sha256" not in unobserved
    assert "socket_transport_observed" not in unobserved


def test_the_observed_route_is_read_from_the_log_and_not_from_the_setting() -> None:
    """A disabled fabric that the log shows in use is still an observed fabric."""

    socket = _route(_SOCKET_LOG, transport="socket")
    assert socket["route"] == "socket"
    assert socket["socket_transport_observed"] is True
    assert socket["infiniband_transport_observed"] is False
    assert socket["evidence_level"] == "observed_debug_log"
    assert isinstance(socket["debug_log_sha256"], str)

    fabric = _route(_ROCE_LOG)
    assert fabric["route"] == "infiniband"
    assert fabric["roce_transport_observed"] is True
    assert fabric["gpu_direct_observed"] is True
    assert fabric["configured_interface_observed"] is True

    # A log naming both means both ran; the fabric is the answer, because a
    # socket channel opening beside an RDMA one does not make the run a socket
    # run. Recording it as a socket run would retract a blocker the evidence
    # does not retract.
    both = _route(_BOTH_ROUTES_LOG)
    assert both["route"] == "infiniband"
    assert both["socket_transport_observed"] is True
    assert both["roce_transport_observed"] is True


def test_a_log_that_does_not_confirm_the_run_is_refused_rather_than_read() -> None:
    """A route the log contradicts is a failed audit, not a narrower one."""

    # A fabric log read against a socket configuration, and the reverse, both
    # mean the run took a route the artifact would not claim.
    with pytest.raises(ClaimBoundaryError):
        _route(_ROCE_LOG, transport="socket")
    with pytest.raises(ClaimBoundaryError):
        _route(_SOCKET_LOG, transport="fabric")

    # A log that names no route at all has shown nothing to check.
    with pytest.raises(ClaimBoundaryError):
        _route("host: rank 0\nsome unrelated line\n")

    # The configured interface has to appear, or the route was not this run's.
    with pytest.raises(ClaimBoundaryError):
        _route(_ROCE_LOG, interface="ens99")


def test_the_infiniband_setting_is_unknown_until_it_is_set() -> None:
    """An unset switch is not the same answer as a switch set to off."""

    assert configured_infiniband_state({}) is None
    assert configured_infiniband_state({"NCCL_IB_DISABLE": "  "}) is None
    assert configured_infiniband_state({"NCCL_IB_DISABLE": "1"}) is True
    assert configured_infiniband_state({"NCCL_IB_DISABLE": "0"}) is False


def test_a_blocker_stands_unless_the_observations_retract_it() -> None:
    """Each retraction needs its own fact, and one fact does not imply another."""

    observed_fabric = {"network": _route(_ROCE_LOG)}
    assert transport_claim_blockers(observed_fabric) == ()
    assert transport_claim_blockers({}) == (BLOCKER_RDMA_NOT_TESTED,)
    assert transport_claim_blockers(
        {"network": _route(_SOCKET_LOG, transport="socket")}
    ) == (BLOCKER_RDMA_NOT_TESTED,)
    # An unread log is `configured_only`, which is not evidence about a route:
    # the run is configured for the fabric and that still does not retract it.
    assert transport_claim_blockers({"network": _route(None)}) == (
        BLOCKER_RDMA_NOT_TESTED,
    )

    # A measurement has to be a measurement: samples, warmups and a positive
    # finite duration each. A key that only says `measured` proves nothing.
    assert measurement_claim_blockers({}) == (
        BLOCKER_PRODUCTION_PERFORMANCE_NOT_MEASURED,
    )
    for incomplete in (
        {"performance": {"measured": True}},
        {"performance": {"measured": True, "synchronized": True}},
    ):
        assert measurement_claim_blockers(incomplete) == (
            BLOCKER_PRODUCTION_PERFORMANCE_NOT_MEASURED,
        )

    # A full gather taken to check an answer is not a production
    # materialization, whether or not it is small.
    gather = gather_claim_blockers(
        production_materialization=False,
        blocker=BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,
    )
    assert gather == (BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,)
    assert (
        gather_claim_blockers(
            production_materialization=True,
            blocker=BLOCKER_VALIDATION_ONLY_TINY_FULL_STATE_GATHER,
        )
        == ()
    )


def test_measured_performance_keeps_every_sample_or_refuses() -> None:
    """Three synchronized samples after a warmup, and the spread is kept."""

    observation = measured_performance(
        [0.002, 0.001, 0.003],
        warmup_iterations=1,
        measurement="sharded_forward",
    )

    assert observation["measured"] is True
    assert observation["synchronized"] is True
    assert observation["measurement"] == "sharded_forward"
    assert observation["warmup_iterations"] == 1
    assert observation["measured_iterations"] == 3
    # Every sample is kept, so a reader can see the spread rather than a summary
    # that would hide it.
    assert observation["seconds"] == [0.002, 0.001, 0.003]
    assert observation["median_seconds"] == pytest.approx(0.002)
    assert observation["minimum_seconds"] == pytest.approx(0.001)
    assert observation["maximum_seconds"] == pytest.approx(0.003)

    for wrong in (
        # Fewer than three samples is not a repeated measurement.
        ([0.001, 0.002], dict(warmup_iterations=1)),
        # No warmup means the first launch is inside the samples.
        ([0.001, 0.002, 0.003], dict(warmup_iterations=0)),
        # A zero or negative duration is not a duration.
        ([0.001, 0.0, 0.003], dict(warmup_iterations=1)),
        ([0.001, -0.002, 0.003], dict(warmup_iterations=1)),
        # A NaN would poison every summary derived from the samples.
        ([0.001, math.nan, 0.003], dict(warmup_iterations=1)),
        ([0.001, math.inf, 0.003], dict(warmup_iterations=1)),
    ):
        samples, options = wrong
        with pytest.raises(ClaimBoundaryError):
            measured_performance(samples, measurement="sharded_forward", **options)


def test_the_staging_blocker_clears_only_for_an_audit_that_found_nothing() -> None:
    """A profiler that did not run has shown nothing; a transfer found is a hit."""

    clear = {
        "host_staging": {
            "profiled": True,
            "host_transfer_observed": False,
            "host_transfer_events": [],
        }
    }
    assert staging_claim_blockers(clear) == ()

    # A profiler that never ran, and one that ran without recording a verdict,
    # both leave the audit incomplete: neither has shown anything about staging.
    for unlooked in (
        {},
        {"host_staging": {"profiled": False}},
        {"host_staging": {"profiled": True}},
    ):
        assert staging_claim_blockers(unlooked) == (
            BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED,
        )

    # A transfer inside the profiled region is a finding. Reporting "nobody
    # looked" there would be false, and dropping the blocker entirely would
    # retract a limitation the audit just confirmed.
    found = staging_claim_blockers(
        {
            "host_staging": {
                "profiled": True,
                "host_transfer_observed": True,
                "host_transfer_events": [{"direction": "device_to_host"}],
            }
        }
    )
    assert found == (BLOCKER_HOST_STAGING_IN_MEASURED_REGION,)
    assert BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED not in found


def test_a_width_or_a_slice_count_counts_only_once_it_crossed_the_hosts() -> None:
    """Two numbers are not a sweep, and an exchange of zero is not one either."""

    assert cut_width_claim_blockers({"1": 32, "2": 64}) == ()
    assert slice_count_claim_blockers({"2": 32, "4": 64}) == ()

    for not_swept in (
        # A single width is one measurement, not a sweep.
        {"1": 32},
        {},
        # A width whose ranks exchanged nothing is a locally reconstructed
        # state: it did not cross the boundary the blocker is about.
        {"1": 0, "2": 0},
        {"1": 32, "2": 0},
    ):
        assert cut_width_claim_blockers(not_swept) == (
            BLOCKER_INTER_NODE_CUT_WIDTH_NOT_SWEPT,
        )
        assert slice_count_claim_blockers(not_swept) == (
            BLOCKER_SLICE_COUNT_FIXED_AT_WORLD_SIZE,
        )


def test_the_union_is_sorted_and_free_of_duplicates() -> None:
    """The artifact's list is a set with a stable order, not a concatenation."""

    union = claim_blockers(
        ("two_node_pair_only_no_wider_topology",),
        (),
        (BLOCKER_RDMA_NOT_TESTED,),
        (BLOCKER_RDMA_NOT_TESTED, BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED),
    )

    assert union == sorted(set(union))
    assert union == [
        BLOCKER_HIDDEN_HOST_STAGING_NOT_AUDITED,
        BLOCKER_RDMA_NOT_TESTED,
        "two_node_pair_only_no_wider_topology",
    ]
