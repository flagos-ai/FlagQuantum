import importlib.util
from pathlib import Path

import torch

SCRIPT = Path("benchmarks/statevector_strong_scaling.py")
SPEC = importlib.util.spec_from_file_location("statevector_strong_scaling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return {
        "schema_version": MODULE.SCHEMA,
        "artifact_class": "measured_development_run",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "world_size": 2,
        "node_count": 2,
        "rank_timings": [{"rank": 0}, {"rank": 1}],
        "rank_placement": [
            {"rank": 0, "hostname": "node-a"},
            {"rank": 1, "hostname": "node-b"},
        ],
        "rank_peak_memory_bytes": [100, 100],
        "communication_tiers": {
            "route_classification": "inter_node_collective",
            "inter_node_logical_bytes_per_rank_max": 100,
        },
        "timing": {"sample_count": 3},
        "correctness": {"passed": True},
    }


def test_workload_is_deterministic_and_world_size_independent():
    first = MODULE.build_workload(8, 3, torch.device("cpu")).to_ir().to_dict()
    second = MODULE.build_workload(8, 3, torch.device("cpu")).to_ir().to_dict()

    assert first == second
    assert len(first["instructions"]) == 24


def test_contract_accepts_complete_development_payload():
    assert MODULE.validate_payload(_payload()) == ()


def test_contract_fails_closed_for_claims_and_incomplete_rank_evidence():
    payload = _payload()
    payload["scalability_claim_allowed"] = True
    payload["release_gate_allowed"] = True
    payload["rank_timings"] = [{"rank": 0}]
    payload["correctness"] = {"passed": False}

    blockers = MODULE.validate_payload(payload)

    assert "development_artifact_must_reject_scalability_claim" in blockers
    assert "development_artifact_must_reject_release" in blockers
    assert "incomplete_rank_timings" in blockers
    assert "correctness_invariants_failed" in blockers


def test_contract_rejects_missing_multinode_placement_and_traffic() -> None:
    payload = _payload()
    payload["rank_placement"] = []
    payload["communication_tiers"] = {}

    blockers = MODULE.validate_payload(payload)

    assert "incomplete_rank_placement" in blockers
    assert "missing_inter_node_communication_evidence" in blockers
