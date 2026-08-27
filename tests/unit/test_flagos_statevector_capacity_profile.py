"""Fail-closed tests for matched FlagOS F5 capacity evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from flagquantum.runtime.distributed.capacity_profile import (
    CAPACITY_RUN_SCHEMA,
    FlagOSCapacityProfileError,
    build_flagos_statevector_capacity_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.distributed_cpu]

WORKLOAD = {
    "name": "flagos_statevector_capacity_f5",
    "n_wires": 12,
    "dtype": "complex128",
}
WORKLOAD_SHA256 = hashlib.sha256(
    json.dumps(WORKLOAD, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _rank(rank: int, *, status: str, sharded: bool = False) -> dict[str, object]:
    return {
        "rank": rank,
        "local_rank": rank,
        "device": f"flagos:{rank}",
        "device_type": "flagos",
        "physical_device_name": "development-gpu",
        "total_memory_bytes": 1000,
        "peak_memory_bytes": 800 if sharded else 1000,
        "status": status,
        "oom_observed": not sharded,
        "error": None if sharded else "OutOfMemoryError: out of memory",
        "local_amplitudes": 512 if sharded else 0,
        "local_state_bytes": 8192 if sharded else 0,
        "communication_count": 2 if sharded else 0,
        "communication_bytes": 4096 if sharded else 0,
        "norm_error": 0.0 if sharded else None,
        "validation_method": (
            "single_full_width_forward_with_fp64_global_norm" if sharded else None
        ),
        "tolerance": 2e-11 if sharded else None,
        "elapsed_seconds": 1.0,
    }


def _run(mode: str) -> dict[str, object]:
    world_size = 1 if mode == "single" else 8
    sharded = mode == "sharded"
    status = "passed" if sharded else "expected_oom"
    return {
        "schema": CAPACITY_RUN_SCHEMA,
        "mode": mode,
        "status": status,
        "expected_outcome": "completion" if sharded else "capacity_failure",
        "world_size": world_size,
        "local_world_size": world_size,
        "node_count": 1,
        "logical_device_type": "flagos",
        "outer_backend": "not_applicable" if mode == "single" else "flagos",
        "distribution_semantics": {
            "single": "single_device_fast_path",
            "replicated": "replicated_full_state_per_rank",
            "sharded": "sharded_across_ranks",
        }[mode],
        "full_state_materialization": not sharded,
        "workload": WORKLOAD,
        "workload_sha256": WORKLOAD_SHA256,
        "ranks": [
            _rank(rank, status=status, sharded=sharded) for rank in range(world_size)
        ],
        "source_revision": "b" * 40,
        "torch_fl_source_revision": "c" * 40,
        "flagcx_route_verified": False,
        "host_staging_observed": None,
        "communication_claim_allowed": False,
        "scalability_claim_allowed": False,
        "production_support_claim_allowed": False,
        "release_gate_allowed": False,
    }


def _runs():
    return [_run(mode) for mode in ("single", "replicated", "sharded")]


def test_matched_capacity_profile_accepts_only_the_three_way_result():
    payload = build_flagos_statevector_capacity_profile(_runs()).to_dict()
    assert payload["development_capacity_expansion_observed"] is True
    assert payload["single_device_capacity_failure_measured"] is True
    assert payload["replicated_capacity_failure_measured"] is True
    assert payload["sharded_capacity_completion_measured"] is True
    assert payload["inner_communication_route"] == "unattributed"
    assert payload["scalability_claim_allowed"] is False
    assert payload["production_support_claim_allowed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda runs: runs[1].__setitem__("workload_sha256", "d" * 64),
        lambda runs: runs[2].__setitem__("flagcx_route_verified", True),
        lambda runs: runs[0]["ranks"][0].__setitem__("oom_observed", False),
        lambda runs: runs[1]["ranks"].pop(),
        lambda runs: runs[2].__setitem__("full_state_materialization", True),
        lambda runs: runs[2]["ranks"][0].__setitem__("communication_bytes", 0),
        lambda runs: runs[2]["ranks"][0].__setitem__("validation_method", "replay"),
    ),
)
def test_capacity_profile_rejects_mismatch_overclaim_and_weak_evidence(mutation):
    runs = copy.deepcopy(_runs())
    mutation(runs)
    with pytest.raises(FlagOSCapacityProfileError):
        profile = build_flagos_statevector_capacity_profile(runs)
        profile.require_accepted()


def test_capacity_profile_rejects_duplicate_modes():
    runs = _runs()
    runs[2]["mode"] = "replicated"
    with pytest.raises(FlagOSCapacityProfileError, match="modes are incomplete"):
        build_flagos_statevector_capacity_profile(runs)
