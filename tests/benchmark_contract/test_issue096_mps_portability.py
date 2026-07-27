import copy
import json
from pathlib import Path

import pytest

from benchmarks.internal.evidence.mps_portability import (
    attribute_communication_tiers,
)
from flagquantum.testing import (
    MPSPortabilityCertificationError,
    require_mps_portability,
)

ROOT = Path(__file__).parents[2]
PREFLIGHT = ROOT / "benchmarks/development/issue096_mps_portability_preflight.json"
pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not PREFLIGHT.exists(), reason="legacy ISSUE-096 evidence is not present"
    ),
]


def valid_payload():
    lifecycle = {
        name: {"passed": True}
        for name in (
            "rendezvous",
            "network_reachability",
            "timeout",
            "teardown",
            "checkpoint_restart",
            "one_rank_failure",
        )
    }
    lifecycle["one_rank_failure"]["partial_artifact_promoted"] = False
    return {
        "schema": "flagquantum.issue096.mps_portability.v1",
        "evidence_source": "measured_runtime",
        "node_count": 2,
        "world_size": 2,
        "rank_placements": [
            {"rank": 0, "hostname": "node-a", "device_uuid": "GPU-a"},
            {"rank": 1, "hostname": "node-b", "device_uuid": "GPU-b"},
        ],
        "source_snapshot": {
            "source_commit": "a" * 40,
            "environment_sha256": "b" * 64,
            "snapshot_sha256": "c" * 64,
        },
        "evidence_artifacts": {
            "parity": {
                "schema": "flagquantum.issue096.parity.v1",
                "snapshot_sha256": "c" * 64,
                "artifact_sha256": "1" * 64,
            },
            "fault_lifecycle": {
                "schema": "flagquantum.issue096.fault_lifecycle.v1",
                "snapshot_sha256": "c" * 64,
                "artifact_sha256": "2" * 64,
            },
        },
        "environment": {"fabric": "InfiniBand HDR"},
        "communication_tiers": {
            "measured": True,
            "topology_aware_boundary_placement": True,
            "intra_node_bytes": 0,
            "inter_node_bytes": 4096,
        },
        "training": {
            "variable_bond": True,
            "forward": True,
            "backward": True,
            "optimizer": True,
            "full_mps_materialization": False,
            "numerical_parity_passed": True,
            "capacity_parity_passed": True,
            "stability_parity_passed": True,
            "checkpoint_restart_completed": True,
            "parity": {
                "max_value_error": 1e-7,
                "value_tolerance": 1e-5,
                "max_gradient_error": 2e-7,
                "gradient_tolerance": 1e-5,
                "max_parameter_error": 0.0,
                "parameter_tolerance": 1e-6,
            },
        },
        "lifecycle": lifecycle,
        "cleanup": {"orphan_process_count": 0},
        "support_matrix": {
            "tested": [
                {
                    "accelerator_class": "A100",
                    "topology": "SXM NVSwitch",
                    "evidence_source": "measured_runtime",
                    "artifact_sha256": "d" * 64,
                },
                {
                    "accelerator_class": "A100",
                    "topology": "PCIe",
                    "evidence_source": "measured_runtime",
                    "artifact_sha256": "e" * 64,
                },
            ],
            "experimental": [],
            "unsupported": [],
        },
        "blockers": [],
        "release_gate_allowed": True,
    }


def test_current_single_node_preflight_is_explicitly_non_promotable():
    payload = json.loads(PREFLIGHT.read_text())
    assert payload["available_node_count"] == 1
    assert payload["release_gate_allowed"] is False
    assert payload["blockers"]
    with pytest.raises(MPSPortabilityCertificationError):
        require_mps_portability(payload)


def test_complete_measured_contract_is_accepted():
    assert require_mps_portability(valid_payload())["release_gate_allowed"] is True


@pytest.mark.parametrize(
    "fault",
    (
        "node",
        "placement",
        "inter_bytes",
        "parity",
        "failure",
        "cleanup",
        "hardware",
        "checkpoint",
        "fabric",
        "provenance",
    ),
)
def test_portability_gate_rejects_partial_or_inferred_evidence(fault):
    payload = copy.deepcopy(valid_payload())
    if fault == "node":
        payload["node_count"] = 1
    elif fault == "placement":
        payload["rank_placements"][1]["hostname"] = "node-a"
    elif fault == "inter_bytes":
        payload["communication_tiers"]["inter_node_bytes"] = 0
    elif fault == "parity":
        payload["training"]["numerical_parity_passed"] = False
    elif fault == "failure":
        payload["lifecycle"]["one_rank_failure"]["passed"] = False
    elif fault == "cleanup":
        payload["cleanup"]["orphan_process_count"] = 1
    elif fault == "hardware":
        payload["support_matrix"]["tested"].pop()
    elif fault == "checkpoint":
        payload["training"]["checkpoint_restart_completed"] = False
    elif fault == "fabric":
        payload["environment"]["fabric"] = "unspecified"
    else:
        payload["support_matrix"]["tested"][1]["artifact_sha256"] = ""
    with pytest.raises(MPSPortabilityCertificationError):
        require_mps_portability(payload)


def test_support_matrix_has_disjoint_status_classes():
    matrix = json.loads(PREFLIGHT.read_text())["support_matrix"]
    assert set(matrix) == {"tested", "experimental", "unsupported"}
    assert all(matrix[name] for name in matrix)


def test_communication_tiers_count_boundary_once_at_compute_owner():
    placements = [
        {"rank": 0, "hostname": "node-a"},
        {"rank": 1, "hostname": "node-b"},
    ]
    update = {
        "compute_owner": 0,
        "communication_peer": 1,
        "payload_bytes": 4096,
    }
    summaries = [
        {"rank": 0, "step_metrics": [{"bond_updates": [update]}]},
        {"rank": 1, "step_metrics": [{"bond_updates": [update]}]},
    ]
    assert attribute_communication_tiers(summaries, placements) == (0, 4096)
