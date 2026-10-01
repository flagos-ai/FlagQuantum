from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from flagquantum.runtime.audit.release_policy import audit_distributed_scalability

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks" / "build_multinode_performance_claim.py"
SPEC = importlib.util.spec_from_file_location(
    "multinode_performance_claim", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
# The builder defines dataclasses, whose annotations the dataclass machinery
# resolves through `sys.modules`, so the module must be registered before it runs.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.fixture()
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A writable copy of the recorded artifacts, so mutations stay local."""

    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    return tmp_path


def _evidence(tree: Path, name: str) -> dict:
    path = tree / f"artifacts/cuda_multinode_{name}_a800_jp171_jp172_20260930.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite(tree: Path, name: str, mutate) -> None:
    """Apply ``mutate`` to one artifact and re-seal its evidence digest."""

    path = tree / f"artifacts/cuda_multinode_{name}_a800_jp171_jp172_20260930.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload["evidence"])
    payload["evidence_sha256"] = MODULE._canonical_sha256(payload["evidence"])
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_committed_claim_matches_what_the_recorded_evidence_implies() -> None:
    committed = (ROOT / MODULE.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    assert committed == MODULE.serialize(MODULE.build_payload())


def test_committed_claim_carries_one_revision_for_all_three_workloads() -> None:
    payload = json.loads((ROOT / MODULE.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["commit"]
    assert {row["source_revision"] for row in payload["workloads"]} == {
        payload["commit"]
    }
    assert {row["route"] for row in payload["workloads"]} == {MODULE.OBSERVED_ROUTE}
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False


def test_the_claim_is_auditable_non_release_evidence() -> None:
    """The artifact must survive the result audit instead of escaping it."""

    payload = json.loads((ROOT / MODULE.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    audit = audit_distributed_scalability(payload)

    assert payload["benchmark_evidence_class"] == "local_non_release"
    assert payload["claim_evidence_type"] == "development_smoke"
    assert payload["non_release_evidence"] is True
    assert payload["distribution_semantics"] == MODULE.DISTRIBUTION_SEMANTICS
    assert payload["scalability_blockers"]
    assert "artifact_class" not in payload
    assert audit.valid, audit.errors
    assert audit.scalability_claim_allowed is False
    assert audit.release_gate_allowed is False


def test_the_claim_reports_the_rank_layout_it_measured() -> None:
    payload = json.loads((ROOT / MODULE.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    world_size = payload["world_size"]

    assert payload["local_world_size"] == 1
    assert payload["node_count"] == world_size
    assert [shard["rank"] for shard in payload["rank_shards"]] == list(
        range(world_size)
    )
    for shard in payload["rank_shards"]:
        evidence = [row["rank_evidence"][shard["rank"]] for row in payload["workloads"]]
        assert shard["local_memory_bytes"] == max(
            item["local_memory_bytes"] for item in evidence
        )
        assert shard["communication_bytes"] == sum(
            item["communication_bytes"] for item in evidence
        )
        assert shard["node_rank"] == shard["rank"]
        assert all(item["owned_work"] for item in evidence)
    assert payload["communication_bytes"] == sum(
        shard["communication_bytes"] for shard in payload["rank_shards"]
    )


def test_the_claim_reports_the_ownership_each_artifact_records() -> None:
    """Ownership comes from the rank records, including a symmetric split."""

    payload = json.loads((ROOT / MODULE.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in payload["workloads"]}

    for row in payload["workloads"]:
        owned = [item["owned_work"] for item in row["rank_evidence"]]
        assert len(owned) == payload["world_size"]
        assert all(owned)
    # Statevector and MPS record which amplitudes and sites each rank owns, so
    # their ranks must name different work. The tensor-network artifacts record
    # one global slice partition that splits evenly, so its rows agree.
    for name in ("statevector", "mps"):
        owned = [item["owned_work"] for item in rows[name]["rank_evidence"]]
        assert len(set(owned)) == len(owned), name
    assert {item["owned_work"] for item in rows["tensor_network"]["rank_evidence"]} == {
        "2 of 4 slice tasks"
    }


def test_the_claim_is_refused_when_a_run_did_not_measure(tree: Path) -> None:
    _rewrite(
        tree,
        "statevector",
        lambda e: e["observations"]["performance"].update(measured=False),
    )
    with pytest.raises(MODULE.ClaimBuildError, match="did not measure"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_samples_were_not_synchronized(
    tree: Path,
) -> None:
    _rewrite(
        tree,
        "mps",
        lambda e: e["observations"]["performance"].update(synchronized=False),
    )
    with pytest.raises(MODULE.ClaimBuildError, match="not synchronized"):
        MODULE.build_payload()


def test_the_claim_is_refused_without_a_warmup(tree: Path) -> None:
    def mutate(evidence: dict) -> None:
        evidence["observations"]["performance"]["warmup_iterations"] = 0

    _rewrite(tree, "tn", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match="warmup iteration"):
        MODULE.build_payload()


def test_the_claim_is_refused_with_too_few_samples(tree: Path) -> None:
    def mutate(evidence: dict) -> None:
        performance = evidence["observations"]["performance"]
        performance["seconds"] = performance["seconds"][:2]
        performance["measured_iterations"] = 2

    _rewrite(tree, "statevector", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match="at least 3 samples"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_summary_disagrees_with_the_samples(
    tree: Path,
) -> None:
    def mutate(evidence: dict) -> None:
        evidence["observations"]["performance"]["minimum_seconds"] = 0.0

    _rewrite(tree, "mps", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match="extremes disagree"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_route_fell_back_to_sockets(tree: Path) -> None:
    def mutate(evidence: dict) -> None:
        network = evidence["observations"]["network"]
        network["route"] = "socket"
        network["socket_transport_observed"] = True

    _rewrite(tree, "statevector", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match="observed route is 'socket'"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_route_was_not_observed(tree: Path) -> None:
    def mutate(evidence: dict) -> None:
        evidence["observations"]["network"]["evidence_level"] = "configured_only"

    _rewrite(tree, "tn", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match="not observed evidence"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_measurement_blocker_survives(tree: Path) -> None:
    def mutate(evidence: dict) -> None:
        evidence["claim_blockers"] = sorted(
            [*evidence["claim_blockers"], MODULE.CONTRADICTED_BLOCKER]
        )

    _rewrite(tree, "mps", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match=MODULE.CONTRADICTED_BLOCKER):
        MODULE.build_payload()


@pytest.mark.parametrize("flag", ("scalability_claim_allowed", "release_gate_allowed"))
def test_the_claim_is_refused_when_the_evidence_claims_more_than_it_did(
    tree: Path, flag: str
) -> None:
    _rewrite(tree, "statevector", lambda e: e.update({flag: True}))
    with pytest.raises(MODULE.ClaimBuildError, match=flag):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_evidence_digest_does_not_match(
    tree: Path,
) -> None:
    path = tree / "artifacts/cuda_multinode_tn_a800_jp171_jp172_20260930.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    copied = copy.deepcopy(payload)
    payload["evidence_sha256"] = "0" * 64
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert copied["evidence_sha256"] != payload["evidence_sha256"]
    with pytest.raises(MODULE.ClaimBuildError, match="digest does not match"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_the_workloads_name_different_revisions(
    tree: Path,
) -> None:
    def mutate(evidence: dict) -> None:
        evidence["environment"]["source_revision"] = "0" * 40

    _rewrite(tree, "mps", mutate)
    with pytest.raises(MODULE.ClaimBuildError, match="different source revisions"):
        MODULE.build_payload()


def test_the_claim_is_refused_when_a_run_did_not_pass(tree: Path) -> None:
    _rewrite(tree, "tn", lambda e: e.update(status="failed"))
    with pytest.raises(MODULE.ClaimBuildError, match="did not pass"):
        MODULE.build_payload()


def test_check_mode_reports_a_missing_or_stale_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    assert MODULE.main(["--check", "--output", MODULE.DEFAULT_OUTPUT]) != 0
    path = tmp_path / MODULE.DEFAULT_OUTPUT
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    assert MODULE.main(["--check", "--output", MODULE.DEFAULT_OUTPUT]) != 0


def test_check_mode_accepts_the_committed_artifact() -> None:
    assert MODULE.main(["--check", "--output", MODULE.DEFAULT_OUTPUT]) == 0
