import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.internal.evidence.statevector_release_gate import (
    RESULTS,
    baseline_results,
    evaluate_issue044_release,
    load_manifest,
)
from benchmarks.internal.evidence.statevector_release_gate import main as gate_main
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
)
from tools.promote_release_candidates import main as promote_main

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]


def _artifact(
    world_size: int,
    acceptance_case: str,
    *,
    node_count: int = 1,
) -> dict[str, object]:
    manifest = load_manifest()
    workload_sha256 = manifest["capacity_workload"]["workload_sha256"]
    baseline = world_size == 1
    evidence = {
        "release_gate_allowed": not baseline,
        "acceptance_case": acceptance_case,
        "world_size": world_size,
        "node_count": node_count,
        "workload_sha256": workload_sha256,
        "distribution_semantics": "sharded_across_ranks",
        "speedup": 1.2,
        "speedup_confidence_interval": [1.1, 1.3],
        "scaling_efficiency": 0.6,
        "single_device_oom_observed": acceptance_case == "single_gpu_capacity_failure",
        "capacity_baseline_device": "cuda:0",
        "capacity_failure_reason": "CUDA out of memory",
        "measured_peak_memory_bytes": 68728000512,
        "training_step_count": 2,
        "full_state_materialized": False,
    }
    if not baseline:
        evidence.update(
            {
                "rank_outputs": [0.0] * world_size,
                "rank_gradients": [0.0] * world_size,
                "rank_timings": [1.0] * world_size,
                "rank_peak_memory_bytes": [1024] * world_size,
                "communication_fraction": 0.1,
                "gpu_activity": 0.8,
            }
        )
    return {
        "schema": "flagquantum_runtime_evidence_v1",
        "artifact_class": "measured_production_run",
        "evidence_scope": ("one_gpu_local" if baseline else "scheduled_4_8_gpu_scale"),
        "provenance": {
            "commit": "a" * 40,
            "workload_sha256": workload_sha256,
            "command": ["torchrun"],
            "devices": [f"GPU-{rank}" for rank in range(world_size)],
            "topology": "test topology",
            "rank_mapping": [f"rank={rank}" for rank in range(world_size)],
            "collective_backend": "nccl",
            "warmup": 5,
            "iterations": 30,
            "seeds": [440044],
            "raw_log_sha256": "b" * 64,
            "fallback_events": [],
        },
        "evidence": evidence,
        "integrity": {},
    }


def _passing_artifacts() -> list[dict[str, object]]:
    return [
        _artifact(1, "single_gpu_capacity_failure"),
        _artifact(2, "matched_speed"),
        _artifact(4, "multi_gpu_capacity_completion"),
        _artifact(8, "multinode_correctness", node_count=2),
    ]


def test_manifest_freezes_workloads_thresholds_and_topologies():
    manifest = load_manifest()
    assert manifest["topologies"]["single_node_world_sizes"] == [2, 4, 8]
    assert manifest["topologies"]["multi_node_required"] is True
    assert manifest["speed_workload"]["minimum_speedup"] > 1
    assert manifest["speed_workload"]["measured_steps"] >= 30
    assert manifest["capacity_workload"]["single_gpu_measured_oom_required"] is True
    assert manifest["runtime"]["backend"] == "pytorch_native"
    capacity = manifest["capacity_workload"]
    workload_path = Path(capacity["manifest_path"])
    # The recorded width is the narrowest one that still exhausts one A800: at
    # 32 wires a single device completes, so a 32-wire baseline would falsify
    # the premise the capacity case rests on rather than support it.
    assert capacity["n_wires"] == 33
    assert (
        hashlib.sha256(workload_path.read_bytes()).hexdigest()
        == capacity["workload_sha256"]
    )
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    assert workload["n_wires"] == capacity["n_wires"]
    assert workload["steps"] == capacity["minimum_optimizer_steps"]


def test_manifest_pins_the_frozen_speed_workload_file():
    """The timing protocol has one frozen definition, and this is its digest."""

    speed = load_manifest()["speed_workload"]
    path = Path(speed["manifest_path"])

    assert hashlib.sha256(path.read_bytes()).hexdigest() == speed["workload_sha256"]
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["warmup_steps"] == speed["warmup_steps"]
    assert frozen["measured_steps"] == speed["measured_steps"]
    names = [item["name"] for item in frozen["configurations"]]
    assert speed["acceptance_configuration"] in names
    assert sorted(item["n_wires"] for item in frozen["configurations"]) == sorted(
        speed["qubits"]
    )
    assert sorted(item["depth"] for item in frozen["configurations"]) == sorted(
        speed["depths"]
    )
    assert sorted(
        item["cross_shard_gate_fraction"] for item in frozen["configurations"]
    ) == sorted(speed["cross_shard_gate_fractions"])


def test_single_device_baseline_is_provenance_evidence_not_a_release_world():
    manifest = load_manifest()
    baseline = manifest["capacity_workload"]["single_gpu_baseline"]

    assert baseline["release_gate_payload"] is False
    assert baseline["evidence_scope"] == "one_gpu_local"
    assert 1 not in manifest["topologies"]["single_node_world_sizes"]


def test_baseline_directory_stays_outside_the_promoted_release_directory():
    """The baseline cannot be promoted, so it cannot live in the promoted set."""

    path = baseline_results(load_manifest())

    assert path != RESULTS
    assert RESULTS not in path.parents
    assert path.is_relative_to("benchmarks/results")


def test_a_baseline_directory_under_the_promoted_set_is_rejected():
    manifest = load_manifest()
    manifest["capacity_workload"]["single_gpu_baseline"]["results_path"] = str(
        RESULTS / "statevector_single_gpu_capacity"
    )

    with pytest.raises(ValueError, match="cannot live under"):
        baseline_results(manifest)


def test_single_device_world_cannot_pose_as_a_release_payload():
    artifacts = _passing_artifacts()
    artifacts[0]["evidence"]["release_gate_allowed"] = True

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "single_device_world_cannot_be_a_release_payload" in blockers


def test_baseline_outside_one_gpu_local_scope_cannot_satisfy_the_capacity_case():
    artifacts = deepcopy(_passing_artifacts())
    artifacts[0]["evidence_scope"] = "scheduled_4_8_gpu_scale"

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "missing_single_gpu_measured_oom_artifact" in blockers


def test_baseline_missing_a_declared_field_is_rejected():
    artifacts = deepcopy(_passing_artifacts())
    del artifacts[0]["evidence"]["capacity_baseline_device"]

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "single_gpu_baseline_missing_required_fields" in blockers


def test_empty_release_directory_fails_closed():
    passed, blockers = evaluate_issue044_release([], load_manifest())
    assert passed is False
    assert "missing_release_world_sizes" in blockers


def test_sealed_envelope_field_layout_can_pass_gate():
    passed, blockers = evaluate_issue044_release(_passing_artifacts(), load_manifest())

    assert passed is True
    assert blockers == ()


def test_named_speed_case_cannot_bypass_frozen_statistical_thresholds():
    artifacts = deepcopy(_passing_artifacts())
    artifacts[1]["evidence"]["speedup_confidence_interval"] = [0.99, 1.4]

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "missing_statistically_significant_speedup_artifact" in blockers


def test_flat_unsigned_production_payload_is_rejected():
    artifact = {
        "artifact_class": "measured_production_run",
        "release_gate_allowed": True,
        "world_size": 8,
    }

    passed, blockers = evaluate_issue044_release([artifact], load_manifest())

    assert passed is False
    assert "invalid_or_unsigned_production_artifact" in blockers


def test_malformed_numeric_measurements_fail_closed_without_crashing():
    artifacts = _passing_artifacts()
    artifacts[1]["evidence"]["speedup"] = None
    artifacts[2]["evidence"]["training_step_count"] = "unknown"

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "missing_statistically_significant_speedup_artifact" in blockers
    assert "missing_multi_gpu_capacity_completion_artifact" in blockers


_KEYS = b"issue-044-promotion-test-key"


def _signed(artifact: dict, *, key: bytes = _KEYS) -> dict:
    # The scope an artifact declares has to match the number of devices the
    # provenance lists, so the world size picks it rather than the fixture.
    world_size = int(artifact["evidence"]["world_size"])
    scope = "one_gpu_local" if world_size == 1 else "scheduled_4_8_gpu_scale"
    if world_size == 2:
        scope = "two_gpu_semantic_regression"
    return create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope(scope),
        provenance=RuntimeProvenance(**artifact["provenance"]),
        evidence=artifact["evidence"],
        signing_key=key,
    ).summary()


def _write_candidates(directory: Path, artifacts: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, artifact in enumerate(artifacts):
        (directory / f"candidate-{index}.json").write_text(
            json.dumps(_signed(artifact)), encoding="utf-8"
        )
    return directory


def _candidate_set(
    root: Path, artifacts: list[dict] | None = None
) -> tuple[Path, Path]:
    """Split a candidate set into its payloads and its capacity baseline.

    A campaign seals only sharded payloads into a candidate directory and leaves
    the single-device baseline in a separate declared provenance directory,
    because one device has no ranks to shard across and can never be release
    evidence. These tests lay a set out the same way and name the baseline
    explicitly, so the gate reads the baseline the set was measured against
    rather than whichever one is checked in.
    """

    items = _passing_artifacts() if artifacts is None else artifacts
    baseline = [item for item in items if item["evidence"]["world_size"] == 1]
    payloads = [item for item in items if item["evidence"]["world_size"] != 1]
    assert len(baseline) == 1
    return (
        _write_candidates(root / "candidates", payloads),
        _write_candidates(root / "baseline", baseline),
    )


def _last_report(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_gate_can_evaluate_a_candidate_set_before_promotion(
    tmp_path, capsys, monkeypatch
):
    """Sealing stages into a candidate directory, so the gate has to read one.

    The strict promotion audit refuses to seal while the release directory holds
    promoted JSON, which is exactly why the campaign is staged elsewhere first.
    """

    candidates, baseline = _candidate_set(tmp_path)
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", _KEYS.decode())

    gate_main(["--candidate", str(candidates), "--baseline-directory", str(baseline)])

    report = _last_report(capsys)
    assert report["passed"] is True
    assert report["artifact_count"] == 4
    assert str(candidates) in report["evaluated"]
    assert str(baseline) in report["evaluated"]


def test_the_gate_rejects_a_candidate_set_that_is_not_release_grade(
    tmp_path, capsys, monkeypatch
):
    candidates, baseline = _candidate_set(
        tmp_path,
        [item for item in _passing_artifacts() if item["evidence"]["world_size"] != 4],
    )
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", _KEYS.decode())

    with pytest.raises(SystemExit) as excinfo:
        gate_main(
            ["--candidate", str(candidates), "--baseline-directory", str(baseline)]
        )

    assert excinfo.value.code == 2
    # The set is rejected for exactly what removing a release world removes --
    # that world size, and the capacity-completion case it carried -- and for
    # nothing else: every other requirement is satisfied by the artifacts.
    assert list(_last_report(capsys)["blockers"]) == [
        "missing_release_world_sizes",
        "missing_multi_gpu_capacity_completion_artifact",
    ]


def test_a_failing_candidate_set_is_reported_rather_than_promoted(
    tmp_path, capsys, monkeypatch
):
    candidates, baseline = _candidate_set(
        tmp_path,
        [item for item in _passing_artifacts() if item["evidence"]["world_size"] != 4],
    )
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", _KEYS.decode())
    release = tmp_path / "release"

    with pytest.raises(SystemExit) as excinfo:
        promote_main(
            [
                "--candidate",
                str(candidates),
                "--baseline-directory",
                str(baseline),
                "--release-directory",
                str(release),
            ]
        )

    assert "missing_release_world_sizes" in str(excinfo.value)
    assert not release.exists()


def test_an_unsigned_candidate_set_is_never_promoted(tmp_path, monkeypatch):
    candidates, baseline = _candidate_set(tmp_path)
    monkeypatch.delenv("FQ_EVIDENCE_SIGNING_KEY", raising=False)
    release = tmp_path / "release"

    with pytest.raises(SystemExit) as excinfo:
        promote_main(
            [
                "--candidate",
                str(candidates),
                "--baseline-directory",
                str(baseline),
                "--release-directory",
                str(release),
            ]
        )

    assert "FQ_EVIDENCE_SIGNING_KEY is required" in str(excinfo.value)
    assert not release.exists()


def test_promotion_moves_a_passing_candidate_set_into_the_release_directory(
    tmp_path, monkeypatch
):
    candidates, baseline = _candidate_set(tmp_path)
    promoted = sorted(path.name for path in candidates.glob("*.json"))
    retained = sorted(path.name for path in baseline.glob("*.json"))
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", _KEYS.decode())
    release = tmp_path / "release"

    assert (
        promote_main(
            [
                "--candidate",
                str(candidates),
                "--baseline-directory",
                str(baseline),
                "--release-directory",
                str(release),
            ]
        )
        == 0
    )

    assert promoted
    assert sorted(path.name for path in release.glob("*.json")) == promoted
    # Moved rather than copied: a candidate that stayed put could be promoted twice.
    assert list(candidates.glob("*.json")) == []
    # The baseline is provenance the gate reads, not a release payload: one
    # device shards across nothing, so it stays in its declared directory.
    assert sorted(path.name for path in baseline.glob("*.json")) == retained
