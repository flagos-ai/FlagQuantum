"""The published series distinguishes measured OOMs from completed training."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.audit_results import audit_paths
from flagquantum.runtime.observability.evidence import verify_evidence_artifact
from tools.sanitize_public_evidence import sanitize_text

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/development/mps_capacity_series_20261023"
SOURCE_COMMIT = "4f23154fe9d944ce3f1e801701d59150138b3bd0"


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
def test_every_capacity_point_has_verifiable_source_and_log_provenance(
    world: int,
) -> None:
    artifact = json.loads((DATA / f"envelopes/world-{world}.json").read_text())
    assert verify_evidence_artifact(artifact) == (True, ())
    assert artifact["schema"] == "flagquantum_runtime_evidence_v2"
    assert artifact["integrity"]["algorithm"] == "sha256"
    assert "signature" not in artifact["integrity"]
    provenance, evidence = artifact["provenance"], artifact["evidence"]
    assert provenance["commit"] == SOURCE_COMMIT
    assert len(set(provenance["devices"])) == world
    assert len(provenance["rank_mapping"]) == world
    assert (
        hashlib.sha256((DATA / f"logs/world-{world}.log").read_bytes()).hexdigest()
        == provenance["raw_log_sha256"]
    )
    assert evidence["world_size"] == world
    assert evidence["logical_mps_bytes"] == 1236780012864
    assert evidence["n_sites"] == 131072
    assert evidence["max_bond_dimension"] == 768
    raw_path = DATA / evidence["publication"]["public_measurements_path"]
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == evidence["publication"]["public_measurements_sha256"]
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_measured_allocation_failures_cannot_become_scalability_claims(
    world: int,
) -> None:
    artifact = json.loads((DATA / f"envelopes/world-{world}.json").read_text())
    evidence = artifact["evidence"]
    assert evidence["release_gate_allowed"] is False
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_payload"] is False
    if world == 1:
        assert evidence["single_device_oom_observed"] is True
    else:
        records = evidence["rank_records"]
        assert [record["rank"] for record in records] == list(range(world))
        assert all(record["status"] == "cuda_oom" for record in records)
        assert all(record["training_summary"] is None for record in records)
        assert [record["intended_owned_wires"] for record in records] == [
            [rank * 131072 // world, (rank + 1) * 131072 // world]
            for rank in range(world)
        ]
    report = audit_paths(
        [DATA / f"envelopes/world-{world}.json"],
        require_scalability=True,
        signing_key=b"",
    )
    assert report["claimable_count"] == 0


def test_the_complete_rank_archive_is_lossless_sanitized_and_scientifically_valid() -> (
    None
):
    index = json.loads((DATA / "measurements/world-16-index.json").read_text())
    records = []
    for rank, entry in enumerate(index["rank_records"]):
        compressed = (DATA / entry["path"]).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == entry["sha256"]
        encoded = gzip.decompress(compressed)
        assert hashlib.sha256(encoded).hexdigest() == entry["uncompressed_sha256"]
        assert sanitize_text(encoded.decode()) == encoded.decode()
        record = json.loads(encoded)
        assert record["rank"] == rank
        assert record["completed_steps"] == 1
        assert record["site_ownership"][rank] == list(
            range(rank * 8192, (rank + 1) * 8192)
        )
        assert record["step_metrics"][0]["useful_work_completed"] is True
        assert record["step_metrics"][0]["optimizer_stage"] == "adam"
        assert any(
            item["optimizer_state_local"] for item in record["optimizer_ownership"]
        )
        records.append(record)
    assert len(records) == 16
    document = dict(index["document"], ranks=records)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(encoded).hexdigest() == index["public_document_sha256"]
    artifact = json.loads((DATA / "envelopes/world-16.json").read_text())
    for key, value in document["measurements"].items():
        assert artifact["evidence"][key] == value
    report = audit_paths(
        [DATA / "envelopes/world-16.json"], require_scalability=True, signing_key=b""
    )
    assert report["claimable_count"] == 1
    assert report["records"][0]["status"] == "ok"


def test_success_and_failed_attempts_keep_their_launcher_diagnostics() -> None:
    for attempt in (1, 2, 3):
        report = json.loads(
            (DATA / f"failed-attempts/{attempt}/watchdog.json").read_text()
        )
        assert report["status"] == "failed_closed"
        assert report["all_launchers_exited"] is True
        assert report["cleanup_verified"] is True
    for name in ("transport-check-watchdog.json", "world-16-watchdog.json"):
        report = json.loads((DATA / name).read_text())
        assert report["status"] == "passed"
        assert report["all_launchers_exited"] is True
