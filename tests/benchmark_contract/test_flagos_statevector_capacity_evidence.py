"""Fail-closed checks for the checked-in FlagOS F5 capacity evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flagquantum.runtime.distributed.capacity_profile import (
    build_flagos_statevector_capacity_profile,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.distributed_cpu]

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/flagos_statevector_capacity_f5_a800_20260827.json"


def test_checked_in_f5_artifact_rebuilds_and_passes_the_capacity_gate():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    profile = build_flagos_statevector_capacity_profile(payload["runs"])
    profile.require_accepted()

    assert profile.to_dict() == payload
    assert payload["development_capacity_expansion_observed"] is True
    assert payload["single_device_capacity_failure_measured"] is True
    assert payload["replicated_capacity_failure_measured"] is True
    assert payload["sharded_capacity_completion_measured"] is True
    assert payload["flagcx_route_verified"] is False
    assert payload["inner_communication_route"] == "unattributed"
    assert payload["scalability_claim_allowed"] is False
    assert payload["production_support_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False


def test_checked_in_f5_artifact_preserves_the_exact_measured_boundary():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    runs = {run["mode"]: run for run in payload["runs"]}
    workload = payload["workload"]

    assert workload["n_wires"] == 32
    assert workload["dtype"] == "complex128"
    assert runs["single"]["world_size"] == 1
    assert runs["replicated"]["world_size"] == 8
    assert runs["sharded"]["world_size"] == 8
    assert all(rank["oom_observed"] for rank in runs["single"]["ranks"])
    assert all(rank["oom_observed"] for rank in runs["replicated"]["ranks"])
    assert all(
        rank["local_amplitudes"] * 8 == 2**32
        and rank["norm_error"] <= rank["tolerance"]
        and rank["peak_memory_bytes"] < rank["total_memory_bytes"]
        and rank["validation_method"]
        == "single_full_width_forward_with_fp64_global_norm"
        for rank in runs["sharded"]["ranks"]
    )
