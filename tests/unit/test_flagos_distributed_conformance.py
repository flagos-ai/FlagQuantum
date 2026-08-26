from __future__ import annotations

from dataclasses import replace

import pytest

import flagquantum as fq
from flagquantum.runtime.distributed.conformance import (
    REQUIRED_FLAGOS_COLLECTIVES,
    REQUIRED_FLAGOS_DTYPES,
    FlagOSCollectiveConformanceCheck,
    FlagOSDistributedConformanceReport,
    FlagOSStatevectorConformanceCheck,
)
from flagquantum.runtime.distributed.identity import (
    DistributedIdentityError,
    build_distributed_identity,
)

pytestmark = pytest.mark.unit


def _identity():
    return build_distributed_identity(
        outer_backend="flagos",
        logical_device="flagos:0",
        rank=0,
        world_size=2,
        process_group_initialized=True,
        platform_identity={"provider": "torch_fl", "provider_version": "test"},
    )


def _collective_checks():
    return tuple(
        FlagOSCollectiveConformanceCheck(
            primitive=primitive,
            dtype=dtype,
            passed=True,
            max_abs_error=0.0,
            elapsed_seconds=0.01,
            payload_bytes=16,
            device_type="flagos",
        )
        for dtype in REQUIRED_FLAGOS_DTYPES
        for primitive in REQUIRED_FLAGOS_COLLECTIVES
    )


def _statevector_checks():
    return tuple(
        FlagOSStatevectorConformanceCheck(
            dtype=dtype,
            passed=True,
            max_abs_error=0.0,
            tolerance=1e-5 if dtype == "complex64" else 1e-11,
            local_amplitudes=4,
            total_amplitudes=8,
            distributed_gate_count=2,
            communication_count=2,
            communication_bytes=64,
            device_type="flagos",
        )
        for dtype in REQUIRED_FLAGOS_DTYPES
    )


def _report():
    return FlagOSDistributedConformanceReport(
        identity=_identity(),
        collective_checks=_collective_checks(),
        statevector_checks=_statevector_checks(),
        rank_placement=(
            {"rank": 0, "local_rank": 0, "device_index": 0, "device": "flagos:0"},
            {"rank": 1, "local_rank": 1, "device_index": 1, "device": "flagos:1"},
        ),
        environment={"torch": "test", "torch_fl": "test"},
        world_size=2,
        local_world_size=2,
        node_count=1,
    )


def test_flagos_mechanical_conformance_remains_fail_closed_for_flagcx():
    report = _report()
    report.require_accepted()

    payload = report.to_dict()
    assert payload["status"] == "passed"
    assert payload["mechanical_conformance_accepted"] is True
    assert payload["statevector_workload_conformance_accepted"] is True
    assert payload["flagcx_route_verified"] is False
    assert payload["host_staging_observed"] is None
    assert payload["communication_claim_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert "flagcx_inner_backend_unverified" in payload["blockers"]
    assert "inner_communication_route_unattributed" in payload["blockers"]
    assert "flagcx_provider_identity_unavailable" not in payload["blockers"]
    assert "single_node_conformance_not_scalability_evidence" in payload["blockers"]

    with pytest.raises(DistributedIdentityError, match="not verified"):
        fq.require_verified_flagcx(report.identity)


def test_flagos_conformance_rejects_incomplete_collective_matrix():
    report = replace(_report(), collective_checks=_collective_checks()[:-1])

    assert report.mechanical_conformance_accepted is False
    assert report.to_dict()["status"] == "failed"
    with pytest.raises(RuntimeError, match="mechanical conformance failed"):
        report.require_accepted()


def test_flagos_conformance_rejects_duplicate_collective_evidence():
    checks = _collective_checks()
    report = replace(_report(), collective_checks=checks + (checks[0],))

    assert report.mechanical_conformance_accepted is False


def test_flagos_conformance_serializes_collective_runtime_failure():
    checks = _collective_checks()
    failed = replace(
        checks[0],
        passed=False,
        error="TypeError: complex dtype unsupported",
    )
    report = replace(_report(), collective_checks=(failed, *checks[1:]))

    assert report.mechanical_conformance_accepted is False
    assert report.statevector_workload_conformance_accepted is False
    assert report.to_dict()["collective_checks"][0]["error"] == (
        "TypeError: complex dtype unsupported"
    )


def test_flagos_conformance_distinguishes_reduce_scatter_from_workload_route():
    checks = tuple(
        replace(
            item,
            passed=False,
            error="TypeError: complex dtype unsupported",
        )
        if item.primitive == "reduce_scatter_tensor"
        else item
        for item in _collective_checks()
    )
    report = replace(_report(), collective_checks=checks)
    payload = report.to_dict()

    assert report.mechanical_conformance_accepted is False
    assert report.statevector_workload_conformance_accepted is True
    assert payload["status"] == "failed"
    assert payload["statevector_workload_conformance_accepted"] is True
    assert "complex_reduce_scatter_unavailable" in payload["blockers"]
    assert payload["communication_claim_allowed"] is False


def test_flagos_conformance_rejects_replicated_or_materialized_statevector():
    materialized = replace(_statevector_checks()[0], full_state_materialization=True)
    report = replace(
        _report(), statevector_checks=(materialized, _statevector_checks()[1])
    )

    assert report.mechanical_conformance_accepted is False


def test_flagos_conformance_rejects_duplicate_rank_device_placement():
    report = replace(
        _report(),
        rank_placement=(
            {"rank": 0, "local_rank": 0, "device_index": 0, "device": "flagos:0"},
            {"rank": 1, "local_rank": 1, "device_index": 0, "device": "flagos:0"},
        ),
    )

    assert report.mechanical_conformance_accepted is False
