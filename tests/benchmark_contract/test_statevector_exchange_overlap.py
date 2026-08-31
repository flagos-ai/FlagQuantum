from benchmarks.statevector_exchange_overlap import (
    SCHEMA,
    _paired_improvement,
    validate_overlap_payload,
)


def _payload():
    return {
        "schema_version": SCHEMA,
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "accelerator_overlap_measured": False,
        "correctness": {"passed": True},
        "pipeline": {"sample_count": 3},
        "synchronous": {"sample_count": 3},
        "runtime": {"pipeline_prefetch_count": 4},
    }


def test_gloo_overlap_artifact_is_valid_development_evidence():
    assert validate_overlap_payload(_payload()) == ()


def test_overlap_artifact_fails_closed_on_false_claims_and_missing_evidence():
    payload = _payload()
    payload["scalability_claim_allowed"] = True
    payload["release_gate_allowed"] = True
    payload["correctness"] = {"passed": False}
    payload["pipeline"] = {"sample_count": 1}
    payload["runtime"] = {"pipeline_prefetch_count": 0}

    blockers = validate_overlap_payload(payload)

    assert "development_artifact_must_reject_scalability_claim" in blockers
    assert "development_artifact_must_reject_release" in blockers
    assert "pipeline_sync_mismatch" in blockers
    assert "insufficient_pipeline_samples" in blockers
    assert "pipeline_did_not_prefetch" in blockers


def test_accelerator_overlap_requires_nccl_and_cuda_events():
    payload = _payload()
    payload["accelerator_overlap_measured"] = True
    payload["backend"] = "gloo"

    blockers = validate_overlap_payload(payload)

    assert "accelerator_overlap_requires_nccl" in blockers
    assert "accelerator_overlap_requires_cuda_events" in blockers


def test_paired_improvement_requires_confidence_interval_above_zero():
    clear = _paired_improvement([0.7, 0.72, 0.71], [1.0, 1.01, 1.02])
    noisy = _paired_improvement([0.9, 1.1, 0.9], [1.0, 1.0, 1.0])

    assert clear["benefit_demonstrated"] is True
    assert clear["confidence_interval_seconds"][0] > 0
    assert noisy["benefit_demonstrated"] is False
