import copy
import json
from pathlib import Path

import pytest

from flagquantum.runtime.backends.mps.production import select_mps_crossover_decision
from flagquantum.testing import MPSScalingCertificationError, require_mps_scaling

ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue095_mps_scaling.json"


pytestmark = [
    pytest.mark.benchmark_contract,
    pytest.mark.skipif(
        not ARTIFACT.exists(), reason="legacy ISSUE-095 evidence is not present"
    ),
]


def load():
    return json.loads(ARTIFACT.read_text())


def complete_fixture():
    payload = load()
    payload["environment"].update(
        {
            "source_commit": "a" * 40,
            "environment_sha256": "b" * 64,
        }
    )
    for run in payload["runs"]:
        run["component_seconds"]["svd_qr"] = 0.0
        run["component_seconds"]["communication"] = 0.0
        run["rank_imbalance"] = 0.0
        run["communication_fraction"] = 0.0
    return payload


def test_legacy_scaling_matrix_is_invalidated_until_remeasured():
    with pytest.raises(MPSScalingCertificationError):
        require_mps_scaling(load(), manifest_root=ROOT)


def test_complete_predeclared_scaling_matrix_passes_audit():
    payload = require_mps_scaling(complete_fixture(), manifest_root=ROOT)
    assert len(payload["runs"]) == 24
    assert payload["release_gate_allowed"] is False
    assert payload["audit"]["post_selection_allowed"] is False


@pytest.mark.parametrize("fault", ("missing", "duplicate", "short_samples", "manifest"))
def test_scaling_audit_rejects_cherry_picked_or_incomplete_samples(fault):
    payload = copy.deepcopy(complete_fixture())
    if fault == "missing":
        payload["runs"].pop()
    elif fault == "duplicate":
        payload["runs"].append(copy.deepcopy(payload["runs"][0]))
    elif fault == "short_samples":
        payload["runs"][0]["raw_samples_seconds"].pop()
        payload["runs"][0]["sample_count"] -= 1
    else:
        payload["predeclaration"]["sha256"] = "0" * 64
    with pytest.raises(MPSScalingCertificationError):
        require_mps_scaling(payload, manifest_root=ROOT)


def test_scaling_audit_recomputes_statistics_from_raw_samples():
    payload = complete_fixture()
    payload["runs"][0]["raw_samples_seconds"][0] *= 2
    with pytest.raises(MPSScalingCertificationError, match="statistics do not match"):
        require_mps_scaling(payload, manifest_root=ROOT)


def test_planner_fails_closed_when_performance_gate_is_false():
    payload = load()
    payload["performance_gate"]["passed"] = False
    point = payload["planner_crossover_surface"][0]
    decision = select_mps_crossover_decision(
        payload,
        family=point["family"],
        sites=point["sites"],
        max_bond=point["max_bond"],
        depth=point["depth"],
        boundary_rate=point["boundary_rate"],
        truncation_policy=point["truncation_policy"],
        topology=point["topology"],
    )
    assert decision["decision"] == "local"
    assert decision["reason"] == "scaling_performance_gate_not_certified"


def test_planner_uses_exact_measured_dimensions_and_fails_closed_off_surface():
    payload = load()
    point = payload["planner_crossover_surface"][0]
    decision = select_mps_crossover_decision(
        payload,
        family=point["family"],
        sites=point["sites"],
        max_bond=point["max_bond"],
        depth=point["depth"],
        boundary_rate=point["boundary_rate"],
        truncation_policy=point["truncation_policy"],
        topology=point["topology"],
    )
    assert decision["decision"] in {"local", "distributed"}
    off_surface = select_mps_crossover_decision(
        payload,
        family=point["family"],
        sites=point["sites"] + 1,
        max_bond=point["max_bond"],
        depth=point["depth"],
        boundary_rate=point["boundary_rate"],
        truncation_policy=point["truncation_policy"],
        topology=point["topology"],
    )
    assert off_surface == {
        "decision": "local",
        "world_size": 1,
        "reason": "no_exact_measured_crossover_point",
        "measured": False,
    }


def test_all_four_regions_are_distinct():
    regions = load()["regions"]
    assert set(regions) == {
        "local_faster",
        "distributed_speedup",
        "weak_scaling",
        "capacity_only",
    }
    encoded = [
        json.dumps(item, sort_keys=True)
        for values in regions.values()
        for item in values
    ]
    assert len(encoded) == len(set(encoded))
