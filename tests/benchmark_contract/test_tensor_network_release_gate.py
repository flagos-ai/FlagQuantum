"""The tensor-network release gate is machine-checked against sealed artifacts.

The gate is the only thing standing between a measured tensor-network run and a
release claim, so these tests seal artifacts exactly as
``tools/seal_runtime_evidence.py`` does and require the gate to accept a complete
set and to name the specific thing that is missing from an incomplete one. The
capacity premise is deliberately absent from the frozen manifest, so the tests
also pin that the gate reports the missing premise instead of assuming one.

The envelope carries no capability key, so the tests also pin that a payload
belonging to another capability -- including the statevector payloads the gate
would otherwise read from its own default results directory -- is refused as
foreign evidence rather than read as a set that is one shape blocker from
passing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.internal.evidence.tensor_network_release_gate import (
    _is_tensor_network_evidence,
    baseline_results,
    evaluate_tensor_network_release,
    load_manifest,
)
from flagquantum.runtime.audit.engine import validate_distributed_claim_evidence
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]

KEY = b"tensor-network-release-gate-test-key"
DEVICE = "NVIDIA A800-SXM4-80GB"

ROOT = Path(__file__).resolve().parents[2]
SCALABILITY = ROOT / "benchmarks/results/scalability"
MATCHED_SPEED_PAIR = (
    ROOT
    / "benchmarks/results/smoke/release_candidates"
    / "tensor_network_matched_speed_pair"
    / "tensor_network_matched_speed_pair.json"
)
CLAIM_KEYS = (
    "claim_evidence_type",
    "release_gate_allowed",
    "release_payload",
    "scalability_claim_allowed",
)


def _workload_digest(manifest: dict) -> str:
    declared = manifest["capacity_workload"]["workload_sha256"]
    source = Path(manifest["capacity_workload"]["workload_manifest_path"])
    frozen = (ROOT / source).read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == declared
    return declared


def _provenance(world: int) -> RuntimeProvenance:
    manifest = load_manifest()
    return RuntimeProvenance(
        commit="a" * 40,
        command=(
            "torchrun",
            "--nnodes=2",
            "benchmarks/tensor_network_release_evidence.py",
        ),
        rank_mapping=tuple(
            f"rank={rank}:node=host-{rank % 2}:device_uuid=GPU-{rank:04d}"
            for rank in range(world)
        ),
        devices=tuple(f"GPU-{rank:04d}" for rank in range(world)),
        topology="GPU0 GPU1 SYS",
        collective_backend="nccl",
        workload_sha256=_workload_digest(manifest),
        warmup=2,
        iterations=10,
        seeds=(440044,),
        raw_log_sha256="c" * 64,
        fallback_events=(),
    )


def _seal(evidence: dict, *, world: int, scope: EvidenceScope, release: bool):
    sealed = dict(evidence)
    if release:
        sealed.update(
            {
                "claim_evidence_type": "production_training_benchmark",
                "release_payload": True,
                "scalability_claim_allowed": True,
                "release_gate_allowed": True,
            }
        )
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=scope,
        provenance=_provenance(world),
        evidence=sealed,
        signing_key=KEY,
    )
    return artifact.summary()


def _ownership() -> dict:
    return {"rank:0": ["parameter:0"], "rank:1": ["parameter:1"]}


def _sharded_evidence(*, world: int = 2, node_count: int = 2) -> dict:
    manifest = load_manifest()
    return {
        "acceptance_case": "multi_gpu_capacity_completion",
        "state_mode": "distributed_tensor_network",
        "world_size": world,
        "local_world_size": 1,
        "node_count": node_count,
        "distribution_semantics": "sharded_across_ranks",
        "collective_backend": "nccl",
        "topology_scope": "multi_node_production_transport",
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "cuda:0",
        "capacity_failure_reason": "the frozen capacity workload exhausted one device",
        "training_step_count": 2,
        "optimizer_update_semantics": "sharded_across_ranks",
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "parameter_ownership": _ownership(),
        "gradient_ownership": _ownership(),
        "optimizer_update_ownership": _ownership(),
        "rank_placement": {"rank:0": "node:0/gpu:0", "rank:1": "node:1/gpu:0"},
        "communication_bytes": 4096,
        "inter_node_communication_bytes": 4096,
        "communication_fraction": 0.01,
        "local_memory_bytes_by_rank": [1024, 1024],
        "measured_peak_memory_bytes": 1024,
        "measured_peak_memory_bytes_by_rank": [1024, 1024],
        "rank_outputs": [0.0, 0.0],
        "rank_gradients": [1.0, 1.0],
        "rank_peak_memory_bytes": [1024, 1024],
        "rank_timings": [1.0, 1.0],
        "timings": [1.0, 1.1],
        "gpu_activity": 1.0,
        "full_state_materialized": False,
        "workload_sha256": _workload_digest(manifest),
        "hardware_inventory": [DEVICE],
    }


def _speed_evidence(*, speedup: float, lower: float) -> dict:
    evidence = _sharded_evidence()
    evidence.update(
        {
            "acceptance_case": "matched_speed",
            "speedup": speedup,
            "speedup_confidence_interval": [lower, speedup + 0.5],
            "scaling_efficiency": speedup / 2.0,
        }
    )
    return evidence


def _baseline_evidence() -> dict:
    manifest = load_manifest()
    return {
        "acceptance_case": "single_gpu_capacity_failure",
        "state_mode": "distributed_tensor_network",
        "world_size": 1,
        "local_world_size": 1,
        "node_count": 1,
        "distribution_semantics": "single_device_fast_path",
        "single_device_oom_observed": True,
        "capacity_baseline_device": "cuda:0",
        "capacity_failure_reason": "the frozen capacity workload exhausted one device",
        "measured_peak_memory_bytes": 85093777408,
        "rank_mapping": "rank=0:node=host-0:device_uuid=GPU-0000",
        "workload_sha256": _workload_digest(manifest),
        "hardware_inventory": [DEVICE],
    }


def _manifest_with_premise() -> dict:
    """Return the frozen manifest with the capacity premise declared established.

    The checked-in manifest records the opposite, because no measured shape
    separates a single-device failure from a sharded completion on this pair.
    The gate's capacity path is still tested, so the tests exercise it against a
    manifest that claims a premise the gate then has to verify.
    """

    manifest = copy.deepcopy(load_manifest())
    manifest["capacity_workload"]["premise_established"] = True
    return manifest


def test_the_checked_in_manifest_declares_no_capacity_premise():
    """The frozen manifest must not claim a premise no measurement supports."""

    manifest = load_manifest()
    assert manifest["capacity_workload"]["premise_established"] is False
    ladder = manifest["capacity_workload"]["measured_ladder"]
    assert ladder, "the falsifying measurements must stay in the manifest"
    completed = [item for item in ladder if item["status"] == "completed"]
    assert completed, "a ladder with no completion says nothing"
    for item in completed:
        world_one = [entry for entry in completed if entry["world_size"] == 1]
        assert world_one, (
            "every measured completion must be compared against one device, "
            "because a premise needs a single-device failure to exist"
        )


def test_the_gate_reports_the_missing_premise_rather_than_assuming_one():
    """Without a premise the gate names it, and does not blame the artifacts."""

    manifest = load_manifest()
    artifacts = [
        _seal(
            _sharded_evidence(),
            world=2,
            scope=EvidenceScope.TWO_GPU_SEMANTIC,
            release=True,
        )
    ]
    passed, blockers = evaluate_tensor_network_release(
        artifacts, manifest, signing_key=KEY
    )
    assert not passed
    assert "capacity_premise_not_established" in blockers
    assert "missing_sharded_training_ownership" not in blockers
    assert "missing_multinode_correctness_artifact" not in blockers


def test_a_pair_payload_with_a_speedup_and_ownership_clears_the_measured_part():
    """The part of the release a pair can measure is accepted when it is present."""

    manifest = _manifest_with_premise()
    artifacts = [
        _seal(
            _speed_evidence(speedup=1.42, lower=1.11),
            world=2,
            scope=EvidenceScope.TWO_GPU_SEMANTIC,
            release=True,
        )
    ]
    passed, blockers = evaluate_tensor_network_release(
        artifacts, manifest, signing_key=KEY
    )
    assert not passed
    # The premise is declared but no capacity completion and no baseline exist,
    # so exactly those two remain, and nothing about the measured speed or the
    # sharded ownership does.
    assert set(blockers) == {
        "missing_single_gpu_measured_oom_artifact",
        "missing_multi_gpu_capacity_completion_artifact",
    }


def test_a_complete_pair_release_passes():
    """Baseline plus completion plus speed is the whole measured contract."""

    manifest = _manifest_with_premise()
    artifacts = [
        _seal(
            _baseline_evidence(),
            world=1,
            scope=EvidenceScope.ONE_GPU_LOCAL,
            release=False,
        ),
        _seal(
            _sharded_evidence(),
            world=2,
            scope=EvidenceScope.TWO_GPU_SEMANTIC,
            release=True,
        ),
        _seal(
            _speed_evidence(speedup=1.42, lower=1.11),
            world=2,
            scope=EvidenceScope.TWO_GPU_SEMANTIC,
            release=True,
        ),
    ]
    passed, blockers = evaluate_tensor_network_release(
        artifacts, manifest, signing_key=KEY
    )
    assert passed, blockers
    assert blockers == ()


def test_an_unsigned_or_tampered_artifact_is_rejected():
    """A payload whose signature does not verify cannot certify anything."""

    manifest = _manifest_with_premise()
    artifact = _seal(
        _sharded_evidence(), world=2, scope=EvidenceScope.TWO_GPU_SEMANTIC, release=True
    )
    tampered = json.loads(json.dumps(artifact))
    tampered["evidence"]["measured_peak_memory_bytes"] = 1
    passed, blockers = evaluate_tensor_network_release(
        [tampered], manifest, signing_key=KEY
    )
    assert not passed
    assert "invalid_or_unsigned_production_artifact" in blockers


def test_a_world_one_payload_cannot_be_a_release_payload():
    """One device cannot shard one logical workload, whatever it claims."""

    manifest = _manifest_with_premise()
    single = _sharded_evidence(world=1, node_count=1)
    single["local_world_size"] = 1
    artifact = _seal(single, world=1, scope=EvidenceScope.ONE_GPU_LOCAL, release=True)
    passed, blockers = evaluate_tensor_network_release(
        [artifact], manifest, signing_key=KEY
    )
    assert not passed
    assert "single_device_world_cannot_be_a_release_payload" in blockers


def test_a_sharded_world_without_ownership_maps_is_not_a_training_claim():
    """Declared semantics without the maps behind them are an assertion."""

    manifest = _manifest_with_premise()
    evidence = _sharded_evidence()
    evidence["gradient_ownership"] = {}
    artifact = _seal(
        evidence, world=2, scope=EvidenceScope.TWO_GPU_SEMANTIC, release=True
    )
    passed, blockers = evaluate_tensor_network_release(
        [artifact], manifest, signing_key=KEY
    )
    assert not passed
    assert "missing_sharded_training_ownership" in blockers


def test_a_speedup_whose_interval_still_contains_one_does_not_qualify():
    """A ratio that could be 1.0 is not a measured speedup."""

    manifest = _manifest_with_premise()
    artifact = _seal(
        _speed_evidence(speedup=1.08, lower=0.94),
        world=2,
        scope=EvidenceScope.TWO_GPU_SEMANTIC,
        release=True,
    )
    passed, blockers = evaluate_tensor_network_release(
        [artifact], manifest, signing_key=KEY
    )
    assert not passed
    assert "missing_statistically_significant_speedup_artifact" in blockers


def _statevector_evidence(*, speedup: float = 1.42, lower: float = 1.11) -> list[dict]:
    """Return the three payloads of a complete set that belongs to another lane.

    Nothing about the shape of the release is missing: a baseline, a sharded
    completion and a statistically significant speedup are all present and named
    the way this contract names them. Only the state mode identifies the
    capability the numbers were measured for, which is what makes the set the
    decisive case for capability admission.
    """

    baseline = _baseline_evidence()
    completion = _sharded_evidence()
    speed = _speed_evidence(speedup=speedup, lower=lower)
    for evidence in (baseline, completion, speed):
        evidence["state_mode"] = "distributed_statevector"
    return [
        _seal(baseline, world=1, scope=EvidenceScope.ONE_GPU_LOCAL, release=False),
        _seal(completion, world=2, scope=EvidenceScope.TWO_GPU_SEMANTIC, release=True),
        _seal(speed, world=2, scope=EvidenceScope.TWO_GPU_SEMANTIC, release=True),
    ]


def _scalability_lane_envelopes() -> list[dict]:
    """Return the sealed payloads the scalability lane already holds.

    The lane is named for the payload rather than for the capability: the five
    checked-in files were sealed by the statevector release campaign, and the
    tensor-network gate reads that directory as its default result location. They
    are read without a signing key here because the capability test is what is
    under test, not the signature.
    """

    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(SCALABILITY.glob("statevector_*.json"))
    ]


def test_a_complete_payload_of_another_capability_is_refused_by_capability():
    """A capability is not established by a payload's shape.

    The set below satisfies every measured requirement this contract states and
    names itself as statevector evidence. A gate that reads it as tensor-network
    evidence has certified the wrong capability.
    """

    manifest = _manifest_with_premise()
    passed, blockers = evaluate_tensor_network_release(
        _statevector_evidence(), manifest, signing_key=KEY
    )
    assert not passed
    assert "production_artifact_is_not_tensor_network_evidence" in blockers


def test_the_statevector_payloads_in_the_scalability_lane_are_not_tn_evidence():
    """The lane's real payloads must not read as an almost-passing empty set.

    Without the capability test these five signed statevector envelopes satisfy
    the release world, multi-node, sharded-training and speed requirements, so
    only the two tensor-network shape blockers remain and a directory that holds
    no tensor-network evidence at all looks one measurement away from passing.
    """

    envelopes = _scalability_lane_envelopes()
    assert envelopes, "the statevector lane must hold the payloads this reads"
    manifest = _manifest_with_premise()
    passed, blockers = evaluate_tensor_network_release(
        envelopes, manifest, signing_key=None
    )
    assert not passed
    assert "production_artifact_is_not_tensor_network_evidence" in blockers
    assert set(blockers) != {
        "missing_single_gpu_measured_oom_artifact",
        "missing_multi_gpu_capacity_completion_artifact",
    }


def test_the_gate_admits_exactly_the_state_modes_the_vocabulary_names():
    """The admission test and the audit vocabulary have to agree.

    A capability test with a second copy of the mode names would drift from the
    vocabulary the rest of the audit reads, and the drift would be invisible: the
    gate would refuse real tensor-network evidence, or admit a mode the audit
    calls something else, whichever copy moved first.
    """

    from flagquantum.runtime.audit.vocabulary import (
        MPS_STATE_MODES,
        STATEVECTOR_STATE_MODES,
        TENSOR_NETWORK_STATE_MODES,
    )

    assert (
        frozenset(
            {
                "distributed_tensor_network",
                "jax_sharded_tensor_network",
                "tensor_network",
            }
        )
        == TENSOR_NETWORK_STATE_MODES
    )
    assert TENSOR_NETWORK_STATE_MODES.isdisjoint(
        STATEVECTOR_STATE_MODES | MPS_STATE_MODES
    )
    for mode in TENSOR_NETWORK_STATE_MODES:
        assert _is_tensor_network_evidence({"state_mode": mode})
    for mode in STATEVECTOR_STATE_MODES | MPS_STATE_MODES | {"", "tn", "unknown"}:
        assert not _is_tensor_network_evidence({"state_mode": mode})
    assert not _is_tensor_network_evidence({})


def test_a_tensor_network_payload_of_another_mode_is_not_evidence_for_the_frozen_run():
    """Admission by family is not admission by the frozen execution path.

    A payload may name any mode the vocabulary recognizes as tensor-network while
    the manifest froze the distributed one, and the payload that names another
    mode was measured by another execution path, so it cannot certify this run.
    """

    manifest = _manifest_with_premise()
    mode = manifest["runtime"]["state_mode"]
    assert mode == "distributed_tensor_network"
    evidence = _sharded_evidence()
    evidence["state_mode"] = "tensor_network"
    artifact = _seal(
        evidence, world=2, scope=EvidenceScope.TWO_GPU_SEMANTIC, release=True
    )
    passed, blockers = evaluate_tensor_network_release(
        [artifact], manifest, signing_key=KEY
    )
    assert not passed
    assert "missing_tensor_network_state_mode_evidence" in blockers
    # The payload is still tensor-network evidence, so the foreign blocker is not
    # the one that fires; the frozen contract is what refuses it.
    assert "production_artifact_is_not_tensor_network_evidence" not in blockers


def test_the_baseline_cannot_live_in_the_scalability_directory():
    """A single-device run is never release-grade sharded evidence."""

    manifest = copy.deepcopy(load_manifest())
    manifest["capacity_workload"]["single_gpu_baseline"][
        "results_path"
    ] = "benchmarks/results/scalability/tensor_network_single_gpu_capacity"
    with pytest.raises(ValueError, match="outside"):
        baseline_results(manifest)


def test_a_frozen_workload_that_drifted_from_its_digest_fails_closed(tmp_path):
    """The premise is the file, so a manifest that disagrees with it is refused."""

    manifest = json.loads(
        (ROOT / "benchmarks/manifests/tensor_network_release_v1.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["capacity_workload"]["workload_sha256"] = "0" * 64
    drifted = tmp_path / "manifest.json"
    drifted.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the digest"):
        load_manifest(drifted)


def test_the_builder_reproduces_the_frozen_capacity_workload(tmp_path):
    """The digested file must be rebuildable, or the manifest froze a stray.

    The release manifest identifies the capacity premise by the digest of a
    checked-in workload file. A file nobody can rebuild is a file that cannot be
    reviewed, so the builder that produced it is run here and its output is
    required to be byte-identical to the frozen artifact.
    """

    from benchmarks.build_tn_capacity_workload import build
    from benchmarks.runners.tn.tn_gap_common import write_workload

    manifest = load_manifest()
    capacity = manifest["capacity_workload"]
    frozen = (ROOT / capacity["workload_manifest_path"]).read_bytes()
    built = tmp_path / "workload.json"
    write_workload(build(*capacity["grid"]), built)
    assert built.read_bytes() == frozen
    assert hashlib.sha256(built.read_bytes()).hexdigest() == capacity["workload_sha256"]


def _committed_matched_speed_pair() -> dict:
    """Return the sealed pair the campaign measured on the two hosts."""

    return json.loads(MATCHED_SPEED_PAIR.read_text(encoding="utf-8"))


def _reseal(evidence: dict, provenance_of: dict, envelope: dict) -> dict:
    """Seal evidence over the committed envelope's own provenance and scope.

    The committed artifact is signed with the campaign's key, which no test
    holds, so its signature cannot be re-verified here. Its content hash can be,
    because that hash covers the unsigned envelope: rebuilding the envelope from
    the committed evidence and provenance reproduces the committed
    ``content_sha256`` exactly when nothing in the artifact was edited, and the
    gate's own signing key is used from there on.
    """

    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass(envelope["artifact_class"]),
        evidence_scope=EvidenceScope(envelope["evidence_scope"]),
        provenance=RuntimeProvenance(**provenance_of),
        evidence=evidence,
        signing_key=KEY,
    )
    return artifact.summary()


def test_the_committed_matched_speed_pair_is_the_measurement_it_records():
    """The sealed candidate must describe the run it was sealed from.

    The campaign recovered the frozen acceptance rung by bounding the reverse
    tape, and this is the artifact that records it. Two things have to hold for
    it to be evidence rather than a file: the artifact must be unedited since it
    was signed, and the run it describes must be the sharded two-host training
    step the contract asks for rather than a single-host timing with a claim
    attached.
    """

    envelope = _committed_matched_speed_pair()
    assert set(envelope) == {
        "artifact_class",
        "evidence",
        "evidence_scope",
        "integrity",
        "provenance",
        "schema",
    }
    assert envelope["artifact_class"] == "measured_production_run"
    assert envelope["evidence_scope"] == EvidenceScope.TWO_GPU_SEMANTIC.value
    assert envelope["integrity"]["algorithm"] == "hmac-sha256"
    committed = envelope["integrity"]["content_sha256"]
    rebuilt = _reseal(envelope["evidence"], envelope["provenance"], envelope)
    assert (
        rebuilt["integrity"]["content_sha256"] == committed
    ), "the committed artifact was edited after it was signed"

    evidence = envelope["evidence"]
    assert evidence["acceptance_case"] == "matched_speed"
    assert evidence["state_mode"] == "distributed_tensor_network"
    assert evidence["distribution_semantics"] == "sharded_across_ranks"
    assert evidence["world_size"] == 2
    assert evidence["node_count"] == 2
    assert evidence["topology_scope"] == "multi_node_production_transport"
    assert evidence["training_step_count"] > 0
    ownership = evidence["gradient_ownership"]
    assert ownership, "sharding without gradients is a forward pass"
    assert evidence["checkpoint_budget_bytes"] == 4294967296, (
        "the bound is what makes the frozen acceptance rung runnable, so the "
        "candidate must record it rather than inherit it from a command line"
    )

    provenance_of = envelope["provenance"]
    assert provenance_of["commit"] == "e4ad30ec42607acf95773bd302977574c1e687fa"
    devices = tuple(provenance_of["devices"])
    assert len(devices) == 2, "a two-host pair runs one device per host here"
    assert len(set(devices)) == 2, "one device cannot be both ranks"
    assert len(provenance_of["rank_mapping"]) == 2
    assert provenance_of["warmup"] == 2
    assert provenance_of["iterations"] == 10
    assert provenance_of["fallback_events"] == []
    command = " ".join(provenance_of["command"])
    assert "--checkpoint-budget-bytes 4294967296" in command
    assert provenance_of["collective_backend"] == "nccl"


def test_the_candidate_digest_check_rejects_an_edited_artifact():
    """The digest check above is only worth running if an edit breaks it.

    The gate verifies a signature with a key the campaign holds and this
    repository does not, so the committed artifact is checked by rebuilding its
    envelope and comparing digests. A statistic changed after sealing must move
    that digest, or the check proves nothing about the file.
    """

    envelope = _committed_matched_speed_pair()
    edited = dict(envelope["evidence"])
    edited["speedup"] = 99.0
    rebuilt = _reseal(edited, envelope["provenance"], envelope)
    committed = envelope["integrity"]["content_sha256"]
    assert rebuilt["integrity"]["content_sha256"] != committed


def test_the_committed_matched_speed_pair_is_not_promoted_and_claims_nothing():
    """It is a candidate: it asserts nothing and sits outside the release lane.

    The capability matrix keeps this pair at ``production_supported`` because the
    contract's capacity premise is unestablished, and a candidate that quietly
    asserted the claim would be a release payload in a directory nobody audits as
    one. The gate must therefore refuse it, and refuse it by naming the claim it
    lacks rather than by reading its measurement.
    """

    envelope = _committed_matched_speed_pair()
    evidence = envelope["evidence"]
    present = [key for key in CLAIM_KEYS if key in evidence]
    assert not present, present
    assert not MATCHED_SPEED_PAIR.is_relative_to(SCALABILITY)
    baseline = baseline_results(load_manifest())
    assert baseline.parts[:4] == (
        "benchmarks",
        "results",
        "smoke",
        "release_candidates",
    ), "the declared baseline must stay outside the promoted lane"
    assert not (ROOT / baseline).is_dir(), (
        "the premise's single-device baseline must stay absent until it is "
        "measured; a stray directory here would be read as the baseline"
    )

    passed, blockers = evaluate_tensor_network_release(
        [_reseal(evidence, envelope["provenance"], envelope)],
        load_manifest(),
        signing_key=KEY,
    )
    assert not passed
    assert set(blockers) == {
        "missing_release_world_sizes",
        "missing_multinode_correctness_artifact",
        "missing_sharded_training_ownership",
        "missing_statistically_significant_speedup_artifact",
        "capacity_premise_not_established",
    }, (
        "a claim-free payload is not release evidence, so the four blockers a "
        "claim would answer must still be reported beside the premise"
    )


def test_the_committed_matched_speed_pair_answers_every_blocker_but_the_premise():
    """With the claim it asserts, only the unestablished premise remains.

    This is the campaign's strongest tensor-network result on this pair, and it
    is asserted here from the committed artifact rather than from the prose of a
    capability note: the four claim keys are the only difference between the
    candidate that reports five blockers and the envelope that reports one. The
    claim the test adds is the same one the release contract asks a promoted
    payload to carry, and the strict audit reads the result as a valid sharded
    training claim, so neither the gate nor the audit is being talked past.
    """

    envelope = _committed_matched_speed_pair()
    claiming = dict(envelope["evidence"])
    claiming.update(
        {
            "claim_evidence_type": "production_training_benchmark",
            "release_gate_allowed": True,
            "release_payload": True,
            "scalability_claim_allowed": True,
        }
    )
    passed, blockers = evaluate_tensor_network_release(
        [_reseal(claiming, envelope["provenance"], envelope)],
        load_manifest(),
        signing_key=KEY,
    )
    assert not passed
    assert blockers == ("capacity_premise_not_established",), (
        "the measured half of the contract must be complete on the committed "
        "artifact, and the premise must be the only thing left"
    )

    audit = validate_distributed_claim_evidence(claiming).summary()
    assert audit["valid"] is True, audit["errors"]
    assert audit["errors"] == ()
    assert audit["distribution_semantics"] == "sharded_across_ranks"
    assert audit["scalability_claim_allowed"] is True
    assert audit["release_gate_allowed"] is True


def test_the_candidate_reports_the_speedup_the_gate_requires():
    """The interval must exclude the speedup the contract refuses to accept.

    The gate reads three numbers and ignores every other statistic in the
    payload, so the ones it reads are checked against the manifest here rather
    than trusted to have been copied correctly.
    """

    manifest = load_manifest()
    speed = manifest["speed_workload"]
    evidence = _committed_matched_speed_pair()["evidence"]
    assert evidence["acceptance_configuration"] == speed["acceptance_configuration"]
    speedup = evidence["speedup"]
    lower, upper = evidence["speedup_confidence_interval"]
    assert lower > speed["confidence_interval_must_exclude_speedup"]
    assert speedup >= speed["minimum_speedup"]
    assert lower <= speedup <= upper
    assert evidence["scaling_efficiency"] >= speed["minimum_scaling_efficiency"]

    accept = [
        rung
        for rung in evidence["configurations"]
        if rung["name"] == speed["acceptance_configuration"]
    ]
    assert len(accept) == 1, "the acceptance rung must be measured, not averaged in"
    rung = accept[0]
    assert rung["speedup"] == speedup
    assert rung["speedup_confidence_interval"] == [lower, upper]
    assert rung["steps"] == 1, (
        "each rung is one full forward-and-reverse step, which is the timing "
        "unit the manifest freezes"
    )
    assert rung["slice_count"] == speed["slice_label_counts"][-1]

    prefix = speed["acceptance_configuration"].split("_18q_")[0] + "_"
    frozen_rungs = {
        f"{prefix}{qubits}q_l{layers}_s{labels}"
        for qubits, layers, labels in zip(
            speed["qubits"], speed["layers"], speed["slice_label_counts"], strict=True
        )
    }
    assert {rung["name"] for rung in evidence["configurations"]} == frozen_rungs
    for rung in evidence["configurations"]:
        assert rung["speedup_confidence_interval"][0] > 1.0, rung
        assert rung["sharded_median_seconds"] < rung["baseline_median_seconds"], rung
        assert rung["sharded_collective_seconds"] > 0.0, rung
