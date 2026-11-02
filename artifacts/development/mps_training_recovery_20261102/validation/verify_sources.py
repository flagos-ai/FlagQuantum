"""Recompute source pins, recovery, ownership and fixed-workload cost reports."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.build_mps_training_recovery_report import report, trajectory

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, timeout=20
    )


def audit() -> dict[str, Any]:
    capacity = json.loads((EVIDENCE / "validation/capacity-leg.json").read_text())
    assert (
        hashlib.sha256(
            git_bytes(
                capacity["repository_snapshot"]["source_commit"], capacity["path"]
            )
        ).hexdigest()
        == capacity["sha256"]
    )
    publication = json.loads(
        (EVIDENCE / "validation/publication-inventory.json").read_text()
    )
    for path, expected in publication["published_sha256"].items():
        assert hashlib.sha256((EVIDENCE / path).read_bytes()).hexdigest() == expected
    renderer = json.loads((EVIDENCE / "validation/report-producer.json").read_text())
    for path, expected in renderer["producer_sha256"].items():
        assert (
            hashlib.sha256(git_bytes(renderer["source_commit"], path)).hexdigest()
            == expected
        )
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    campaigns = []
    for name, worlds in (("cpu", [1, 2]), ("cuda", [1, 2, 4, 8, 16])):
        directory = EVIDENCE / name
        if not directory.exists():
            continue
        payload, markdown = report(directory, worlds)
        assert payload == json.loads((directory / "summary.json").read_text())
        assert markdown == (directory / "REPORT.md").read_text()
        source = payload["source"]
        commit = source["code_version"]
        assert source["dirty"] is False
        for path, expected in source["producer_sha256"].items():
            assert hashlib.sha256(git_bytes(commit, path)).hexdigest() == expected
        manifest = git_bytes(
            commit, "benchmarks/manifests/mps_training_recovery_v1.json"
        )
        assert hashlib.sha256(manifest).hexdigest() == source["protocol_sha256"]
        assert json.loads(manifest) == payload["protocol"]
        for world in worlds:
            path = directory / f"world-{world}.json"
            raw = (
                path.read_bytes()
                if path.exists()
                else gzip.decompress(path.with_suffix(".json.gz").read_bytes())
            )
            point = json.loads(raw)
            assert point["source"] == source and point["recovery_passed"]
            assert (
                trajectory(point, "parameter_values")[0]
                == payload["protocol"]["initial_parameters"]
            )
            assert point["node_count"] == (2 if world == 16 else 1)
            assert point["local_world_size"] == min(world, 8)
            uuids = set()
            for record in point["ranks"]:
                rank = record["rank"]
                for step in record["uninterrupted"]["step_metrics"]:
                    assert step["learning_rate"] == payload["protocol"]["learning_rate"]
                    assert step["optimizer_stage"] == "adam"
                    assert (
                        step["largest_realized_bond"] <= payload["protocol"]["max_bond"]
                    )
                if name == "cuda":
                    inventory = json.loads((directory / "inventory.json").read_text())
                    assert record["node_rank"] == rank // 8
                    assert record["local_rank"] == rank % 8
                    observed = record["device_uuid"].removeprefix("GPU-").lower()
                    physical = (
                        inventory["hosts"][record["node_rank"]][record["local_rank"]][
                            "uuid"
                        ]
                        .removeprefix("GPU-")
                        .lower()
                    )
                    assert observed == physical
                    uuids.add(observed)
                for leg, key in (
                    ("uninterrupted", "checkpoint_reference"),
                    ("restarted", "checkpoint_resumed"),
                ):
                    root = directory / f"world-{world}-checkpoints" / leg
                    manifest_path = root / "COMMITTED.json"
                    manifest_data = manifest_path.read_bytes()
                    receipt = record[key]
                    assert (
                        hashlib.sha256(manifest_data).hexdigest()
                        == receipt["manifest_sha256"]
                    )
                    checkpoint_manifest = json.loads(manifest_data)
                    assert checkpoint_manifest["world_size"] == world
                    assert (
                        checkpoint_manifest["completed_steps"]
                        == payload["protocol"]["steps"]
                    )
                    shard = next(
                        s for s in checkpoint_manifest["shards"] if s["rank"] == rank
                    )
                    assert shard["sha256"] == receipt["shard_sha256"]
                    assert (
                        root / (shard["file"] + ".sha256")
                    ).read_text().strip() == receipt["shard_sha256"]
                prefix = record["checkpoint_prefix"]
                prefix_hash = (
                    directory
                    / f"world-{world}-checkpoints/restarted/rank-{rank}-step-{payload['protocol']['checkpoint_step']}.pt.sha256"
                )
                assert prefix_hash.read_text().strip() == prefix["shard_sha256"]
            if name == "cuda":
                assert len(uuids) == world
        for row in payload["rows"]:
            tolerance = payload["protocol"]["comparison_atol"]
            assert row["matched_local_objective_error"] <= tolerance
            assert row["matched_local_projected_gradient_error"] <= tolerance
            assert all(
                value is None or value <= tolerance
                for value in row["scaling_errors_vs_world_1"].values()
            )
        campaigns.append(
            {
                "campaign": name,
                "source_commit": commit,
                "world_sizes": worlds,
                "source_and_manifest_hashes_verified": True,
                "raw_recovery_ownership_and_sharding_errors_recomputed": True,
                "checkpoint_commit_and_owner_receipts_verified": True,
                "reports_reproduce_byte_for_byte": True,
            }
        )
    discovery = json.loads(
        (EVIDENCE / "validation/pre-freeze-debugging.json").read_text()
    )
    assert (
        hashlib.sha256(
            (EVIDENCE / "validation" / discovery["archive"]).read_bytes()
        ).hexdigest()
        == discovery["archive_sha256"]
    )
    return {"campaigns": campaigns}


if __name__ == "__main__":
    payload = audit()
    (EVIDENCE / "validation/source-and-sample-audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
